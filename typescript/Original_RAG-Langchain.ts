import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MemorySaver } from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrieverTool } from "langchain/tools/retriever";
import { Document } from "@langchain/core/documents";
import { HumanMessage } from "@langchain/core/messages";
import dotenv from "dotenv";

dotenv.config({ path: "../shared/.env" });

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  throw new Error("OPENAI_API_KEY not found in environment variables");
}

// Data
const documents = [
  // France
  "The capital of France is Paris.",
  "The Eiffel Tower is located in Paris.",
  "France is known for its wine and cheese.",
  "The French Riviera is a popular tourist destination in France.",
  "Paris is also called the City of Light.",
  "France shares borders with countries like Germany, Italy, and Spain.",
  "The Louvre in France is the world's largest art museum.",
  "Many French people enjoy visiting the USA for its iconic landmarks.",
  "France and the UK are separated by the English Channel.",
  "French cuisine is admired worldwide, especially in Vietnam due to historical ties.",

  // USA
  "The capital of the USA is Washington, D.C.",
  "The Statue of Liberty is located in New York City.",
  "The USA is known for its diverse culture and innovation.",
  "Hollywood in the USA is the center of the global film industry.",
  "The Grand Canyon is a natural wonder in the USA.",
  "The USA and China are two of the largest economies in the world.",
  "Jazz music originated in the USA and influenced cultures worldwide, including Japan.",
  "Many American students study abroad in countries like Germany and the UK.",
  "The USA hosts large Vietnamese communities, especially in California.",
  "Silicon Valley in the USA is a hub for technology and startups.",

  // UK
  "The capital of the UK is London.",
  "Big Ben is located in London.",
  "The UK is known for its tea and royal family.",
  "Oxford University in the UK is one of the oldest universities in the world.",
  "The UK and France share a tunnel called the Channel Tunnel.",
  "Shakespeare, a famous playwright, was born in the UK.",
  "The UK and the USA have a long-standing 'special relationship.'",
  "Football (soccer) originated in the UK and is popular worldwide.",
  "Many British tourists enjoy visiting Italy for its historical landmarks.",
  "The Beatles, a legendary music band, are from the UK.",

  // Germany
  "The capital of Germany is Berlin.",
  "The Brandenburg Gate is located in Berlin.",
  "Germany is known for its cars and beer.",
  "Oktoberfest in Germany is the world's largest beer festival.",
  "Germany is a leading country in renewable energy development.",
  "Germany and France are founding members of the European Union.",
  "Many German philosophers like Kant and Hegel shaped modern thought.",
  "German engineering is admired globally, especially in Japan.",
  "The Autobahn in Germany has sections with no speed limit.",
  "Germany and the USA collaborate on many scientific projects.",

  // Italy
  "The capital of Italy is Rome.",
  "The Colosseum is located in Rome.",
  "Italy is known for its pasta and pizza.",
  "Venice in Italy is famous for its canals and gondolas.",
  "Italy shares borders with France, Switzerland, and Austria.",
  "The Leaning Tower of Pisa is a famous landmark in Italy.",
  "Italian art from the Renaissance is admired in countries like the UK and France.",
  "Italy and Japan have cultural exchange programs involving fashion and design.",
  "Rome is often called the Eternal City.",
  "Many tourists from the USA visit Italy for its rich history and cuisine.",

  // South Korea
  "The capital of South Korea is Seoul.",
  "Gyeongbokgung Palace is located in Seoul.",
  "South Korea is known for its K-pop and technology.",
  "Samsung, a global tech giant, is headquartered in South Korea.",
  "South Korea and the USA are strong allies in East Asia.",
  "Many South Korean dishes, like kimchi, are popular worldwide, including in Vietnam.",
  "The DMZ separates South Korea from North Korea.",
  "South Korea has a rich history of hanbok, its traditional clothing.",
  "Seoul is home to many global conferences, attracting participants from the UK and Germany.",
  "South Korea and Japan have shared cultural influences despite historical tensions.",

  // Japan
  "The capital of Japan is Tokyo.",
  "Mount Fuji is located near Tokyo.",
  "Japan is known for its sushi and anime.",
  "The Shinkansen, or bullet train, is a technological marvel in Japan.",
  "Cherry blossoms in Japan attract visitors from countries like the USA and China.",
  "Japan has a strong cultural influence on South Korea through manga and J-pop.",
  "Tokyo hosted the Summer Olympics in 2021.",
  "Japan and Germany collaborate in automobile manufacturing.",
  "Many Japanese tourists visit Italy for its art and architecture.",
  "Japan's tea ceremonies are similar in cultural significance to China's tea traditions.",

  // China
  "The capital of China is Beijing.",
  "The Great Wall of China is located near Beijing.",
  "China is known for its tea and ancient history.",
  "Shanghai is one of China's most modern and populous cities.",
  "China and Vietnam share a long border and cultural exchanges.",
  "Many Chinese tourists visit France for its luxury goods.",
  "The Silk Road was an ancient trade route connecting China to Europe.",
  "China and the USA have strong trade relations.",
  "Chinese New Year is celebrated worldwide, including in the UK and Germany.",
  "The Terracotta Army is a famous archaeological site in China.",

  // Vietnam
  "The capital of Vietnam is Hanoi.",
  "Hoan Kiem Lake is located in Hanoi.",
  "Vietnam is known for its pho and coffee.",
  "Ha Long Bay in Vietnam is a UNESCO World Heritage Site.",
  "Vietnam shares a border with China and Laos.",
  "Many Vietnamese people immigrated to the USA after the Vietnam War.",
  "The French colonial period influenced Vietnamese architecture and cuisine.",
  "Vietnam and South Korea have strong cultural ties through entertainment and trade.",
  "The Mekong Delta is an important agricultural region in Vietnam.",
  "Vietnamese students often study abroad in countries like Japan and Australia.",
];

/**
 * Chuyển đổi danh sách văn bản thành hệ thống tìm kiếm thông minh
 *
 * @param documents Danh sách các đoạn văn bản cần xử lý
 * @returns {Promise<{retriever: any, vectorstore: any}>} Công cụ tìm kiếm và kho vector
 *
 * Quy trình:
 * 1. Chuyển mỗi đoạn văn bản thành đối tượng Document
 * 2. Chia nhỏ văn bản thành các đoạn dễ xử lý (1000 ký tự/đoạn)
 * 3. Chuyển văn bản thành vector số học để máy tính hiểu được
 * 4. Tạo công cụ tìm kiếm với khả năng trả về 3 kết quả gần nhất
 */
async function getRetriever(documents: string[]) {
  try {
    // Chuyển đổi văn bản thành Document objects
    const docs = documents.map((doc) => new Document({ pageContent: doc }));

    // Khởi tạo text splitter
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const splits = await textSplitter.splitDocuments(docs);

    // Tạo vector store và retriever
    const vectorstore = await MemoryVectorStore.fromDocuments(
      splits,
      new OpenAIEmbeddings()
    );
    const retriever = vectorstore.asRetriever({
      k: 3,
    });

    return { retriever, vectorstore };
  } catch (e) {
    console.error(`❌ Error when creating retriever: ${e}`);
    // Trả về retriever mặc định khi có lỗi
    const defaultDoc = [
      new Document({
        pageContent: "Error when creating retriever. Please try again later.",
        metadata: { source: "error" },
      }),
    ];
    const vectorstore = await MemoryVectorStore.fromDocuments(
      defaultDoc,
      new OpenAIEmbeddings()
    );
    return { retriever: vectorstore.asRetriever(), vectorstore };
  }
}

/**
 * Thực hiện tìm kiếm thông minh và hiển thị quá trình suy luận của AI
 *
 * @param query Câu hỏi người dùng muốn tìm câu trả lời
 * @param llm Mô hình AI để xử lý ngôn ngữ tự nhiên
 * @param retriever Công cụ tìm kiếm thông minh đã được tạo
 * @param memory Bộ nhớ để lưu lại quá trình suy luận
 *
 * Quy trình hoạt động:
 * 1. Tạo công cụ tìm kiếm với khả năng tìm thông tin về các quốc gia
 * 2. Khởi tạo AI agent có khả năng suy luận
 * 3. Stream và hiển thị từng bước suy luận của AI
 * 4. Trả về toàn bộ quá trình suy luận và kết quả
 */
async function search(
  query: string,
  llm: ChatOpenAI,
  retriever: any,
  memory: MemorySaver
) {
  // Tạo công cụ tìm kiếm
  const tool = createRetrieverTool(retriever, {
    name: "find",
    description:
      "Search and return information about countries, including: Basic information, landmarks, culture, and international relations.",
  });

  // Khởi tạo agent
  const agentExecutor = createReactAgent({
    llm,
    tools: [tool],
    checkpointSaver: memory,
  });

  const config = {
    configurable: {
      thread_id: "abc123",
    },
  };

  try {
    console.log("\n🤔 Processing question:", query);

    const stream = await agentExecutor.stream(
      { messages: [new HumanMessage(query)] },
      config
    );

    let finalResponse = null;
    for await (const event of await stream) {
      console.log("event", event);
      if (event.agent?.messages?.[0]) {
        finalResponse = event.agent.messages[0].content;
      }
    }

    return finalResponse || "No answer found.";
  } catch (e) {
    console.error(`❌ Error when searching: ${e}`);
    return "Error when searching. Please try again later.";
  }
}

/**
 * Hàm chính của chương trình
 *
 * Các bước thực hiện:
 * 1. Tạo bộ nhớ để lưu quá trình suy luận
 * 2. Khởi tạo mô hình AI GPT-4 với độ sáng tạo = 0 (để có câu trả lời nhất quán)
 * 3. Tạo hệ thống tìm kiếm từ dữ liệu có sẵn
 * 4. Thực hiện tìm kiếm với câu hỏi mẫu về văn hóa Việt Nam
 */
async function main() {
  try {
    const memory = new MemorySaver();
    const llm = new ChatOpenAI({ model: "gpt-4", temperature: 0 });
    const { retriever } = await getRetriever(documents);

    const query = "I am learning about Vietnamese culture.";
    const result = await search(query, llm, retriever, memory);
    console.log("\n📝 Answer:", result);
  } catch (error) {
    console.error("❌ Error:", error);
  }
}

// Chạy chương trình
main();
