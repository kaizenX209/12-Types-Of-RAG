import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Annotation } from "@langchain/langgraph";
import { createRetrieverTool } from "langchain/tools/retriever";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { END } from "@langchain/langgraph";
import { pull } from "langchain/hub";
import { z } from "zod";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { AIMessage, BaseMessage } from "@langchain/core/messages";
import { StateGraph } from "@langchain/langgraph";
import { START } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

import * as dotenv from "dotenv";

const GraphState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
});

async function loadAndProcessDocuments(pdfPath: string) {
  const loader = new PDFLoader(pdfPath);
  const docs = await loader.load();

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });
  const docSplits = await textSplitter.splitDocuments(docs);

  // Add to vectorDB
  const vectorStore = await MemoryVectorStore.fromDocuments(
    docSplits,
    new OpenAIEmbeddings()
  );

  return vectorStore.asRetriever();
}

function shouldRetrieve(state: typeof GraphState.State): string {
  const { messages } = state;
  console.log("---DECIDE TO RETRIEVE---");
  const lastMessage = messages[messages.length - 1];

  if (
    "tool_calls" in lastMessage &&
    Array.isArray(lastMessage.tool_calls) &&
    lastMessage.tool_calls.length
  ) {
    console.log("---DECISION: RETRIEVE---");
    return "retrieve";
  }
  // If there are no tool calls then we finish.
  return END;
}

/**
 * Xác định xem Agent có nên tiếp tục dựa trên mức độ liên quan của các tài liệu được truy xuất hay không.
 * Hàm này kiểm tra xem tin nhắn cuối cùng trong cuộc hội thoại có phải là FunctionMessage hay không,
 * cho biết việc truy xuất tài liệu đã được thực hiện. Sau đó nó đánh giá mức độ liên quan của các tài liệu này
 * với câu hỏi ban đầu của người dùng bằng cách sử dụng một mô hình và bộ phân tích đầu ra được xác định trước.
 * Nếu tài liệu có liên quan, cuộc hội thoại được coi là hoàn tất. Ngược lại, quá trình truy xuất sẽ được tiếp tục.
 * @param {typeof GraphState.State} state - Trạng thái hiện tại của agent, bao gồm tất cả các tin nhắn.
 * @returns {Promise<Partial<typeof GraphState.State>>} - Trạng thái đã cập nhật với tin nhắn mới được thêm vào danh sách tin nhắn.
 */
async function gradeDocuments(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---GET RELEVANCE---");

  const { messages } = state;
  const tool = {
    name: "give_relevance_score",
    description: "Give a relevance score to the retrieved documents.",
    schema: z.object({
      binaryScore: z.string().describe("Relevance score 'yes' or 'no'"),
    }),
  };

  const prompt = ChatPromptTemplate.fromTemplate(
    `You are a grader assessing relevance of retrieved docs to a user question.
  Here are the retrieved docs:
  \n ------- \n
  {context} 
  \n ------- \n
  Here is the user question: {question}
  If the content of the docs are relevant to the users question, score them as relevant.
  Give a binary score 'yes' or 'no' score to indicate whether the docs are relevant to the question.
  Yes: The docs are relevant to the question.
  No: The docs are not relevant to the question.`
  );

  const model = new ChatOpenAI({
    model: "gpt-4o",
    temperature: 0,
  }).bindTools([tool], {
    tool_choice: tool.name,
  });

  const chain = prompt.pipe(model);

  const lastMessage = messages[messages.length - 1];

  const score = await chain.invoke({
    question: messages[0].content as string,
    context: lastMessage.content as string,
  });

  return {
    messages: [score],
  };
}

/**
 * Kiểm tra mức độ liên quan của lệnh gọi công cụ LLM trước đó.
 *
 * @param {typeof GraphState.State} state - Trạng thái hiện tại của agent, bao gồm tất cả các tin nhắn.
 * @returns {string} - Một chỉ thị "yes" hoặc "no" dựa trên mức độ liên quan của tài liệu.
 */
function checkRelevance(state: typeof GraphState.State): string {
  console.log("---CHECK RELEVANCE---");

  const { messages } = state;
  const lastMessage = messages[messages.length - 1];
  if (!("tool_calls" in lastMessage)) {
    throw new Error(
      "The 'checkRelevance' node requires the most recent message to contain tool calls."
    );
  }
  const toolCalls = (lastMessage as AIMessage).tool_calls;
  if (!toolCalls || !toolCalls.length) {
    throw new Error("Last message was not a function message");
  }

  if (toolCalls[0].args.binaryScore === "yes") {
    console.log("---DECISION: DOCS RELEVANT---");
    return "yes";
  }
  console.log("---DECISION: DOCS NOT RELEVANT---");
  return "no";
}

// Nodes

/**
 * Gọi mô hình agent để tạo phản hồi dựa trên trạng thái hiện tại.
 * Hàm này gọi mô hình agent để tạo phản hồi cho trạng thái hội thoại hiện tại.
 * Phản hồi được thêm vào tin nhắn của trạng thái.
 * @param {typeof GraphState.State} state - Trạng thái hiện tại của agent, bao gồm tất cả các tin nhắn.
 * @returns {Promise<Partial<typeof GraphState.State>>} - Trạng thái đã cập nhật với tin nhắn mới được thêm vào danh sách tin nhắn.
 */
async function agent(
  state: typeof GraphState.State,
  tools: any[]
): Promise<Partial<typeof GraphState.State>> {
  console.log("---CALL AGENT---");

  const { messages } = state;
  // Find the AIMessage which contains the `give_relevance_score` tool call,
  // and remove it if it exists. This is because the agent does not need to know
  // the relevance score.
  const filteredMessages = messages.filter((message) => {
    if (
      "tool_calls" in message &&
      Array.isArray(message.tool_calls) &&
      message.tool_calls.length > 0
    ) {
      return message.tool_calls[0].name !== "give_relevance_score";
    }
    return true;
  });

  const model = new ChatOpenAI({
    model: "gpt-4o",
    temperature: 0,
    streaming: true,
  }).bindTools(tools);

  const response = await model.invoke(filteredMessages);
  return {
    messages: [response],
  };
}

/**
 * Chuyển đổi câu truy vấn để tạo ra một câu hỏi tốt hơn.
 * @param {typeof GraphState.State} state - Trạng thái hiện tại của agent, bao gồm tất cả các tin nhắn.
 * @returns {Promise<Partial<typeof GraphState.State>>} - Trạng thái đã cập nhật với tin nhắn mới được thêm vào danh sách tin nhắn.
 */
async function rewrite(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---TRANSFORM QUERY---");

  const { messages } = state;
  const question = messages[0].content as string;
  const prompt = ChatPromptTemplate.fromTemplate(
    `Look at the input and try to reason about the underlying semantic intent / meaning. \n 
Here is the initial question:
\n ------- \n
{question} 
\n ------- \n
Formulate an improved question:`
  );

  // Grader
  const model = new ChatOpenAI({
    model: "gpt-4o",
    temperature: 0,
    streaming: true,
  });
  const response = await prompt.pipe(model).invoke({ question });
  return {
    messages: [response],
  };
}

/**
 * Tạo câu trả lời
 * @param {typeof GraphState.State} state - Trạng thái hiện tại của agent, bao gồm tất cả các tin nhắn.
 * @returns {Promise<Partial<typeof GraphState.State>>} - Trạng thái đã cập nhật với tin nhắn mới được thêm vào danh sách tin nhắn.
 */
async function generate(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---GENERATE---");

  const { messages } = state;
  const question = messages[0].content as string;
  // Extract the most recent ToolMessage
  const lastToolMessage = messages
    .slice()
    .reverse()
    .find((msg) => msg._getType() === "tool");
  if (!lastToolMessage) {
    throw new Error("No tool message found in the conversation history");
  }

  const docs = lastToolMessage.content as string;

  const prompt = await pull<ChatPromptTemplate>("rlm/rag-prompt");

  const llm = new ChatOpenAI({
    model: "gpt-4o",
    temperature: 0,
    streaming: true,
  });

  const ragChain = prompt.pipe(llm);

  const response = await ragChain.invoke({
    context: docs,
    question,
  });

  return {
    messages: [response],
  };
}

async function main() {
  // Load environment variables
  dotenv.config({ path: "../shared/.env" });

  const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
  if (!OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY not found in environment variables");
  }

  const PdfPath = "../documents/2404.16130v1_Graph-RAG.pdf";
  const retriever = await loadAndProcessDocuments(PdfPath);

  const tool = createRetrieverTool(retriever, {
    name: "retrieve_graph_rag",
    description: "Search and return information about Graph-RAG.",
  });
  const tools = [tool];

  const toolNode = new ToolNode<typeof GraphState.State>(tools);

  // Define the graph
  const workflow = new StateGraph(GraphState)
    .addNode("agent", async (state) => agent(state, tools))
    .addNode("retrieve", toolNode)
    .addNode("gradeDocuments", gradeDocuments)
    .addNode("rewrite", rewrite)
    .addNode("generate", generate);

  // Call agent node to decide to retrieve or not
  workflow.addEdge(START, "agent");

  // Decide whether to retrieve
  workflow.addConditionalEdges(
    "agent",
    // Assess agent decision
    shouldRetrieve
  );

  workflow.addEdge("retrieve", "gradeDocuments");

  // Edges taken after the `action` node is called.
  workflow.addConditionalEdges(
    "gradeDocuments",
    // Assess agent decision
    checkRelevance,
    {
      // Call tool node
      yes: "generate",
      no: "rewrite", // placeholder
    }
  );

  workflow.addEdge("generate", END);
  workflow.addEdge("rewrite", "agent");

  // Compile
  const app = workflow.compile();

  const inputs = {
    messages: [new HumanMessage("What is Graph-RAG?")],
  };
  let finalState;
  for await (const output of await app.stream(inputs)) {
    for (const [key, value] of Object.entries(output)) {
      const lastMsg = output[key].messages[output[key].messages.length - 1];
      console.log(`Output from node: '${key}'`);
      console.dir(
        {
          type: lastMsg._getType(),
          content: lastMsg.content,
          tool_calls: lastMsg.tool_calls,
        },
        { depth: null }
      );
      console.log("---\n");
      finalState = value;
    }
  }

  console.log(JSON.stringify(finalState, null, 2));
}

main().catch(console.error);
