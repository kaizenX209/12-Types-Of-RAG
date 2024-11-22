"""
Graph-RAG Implementation sử dụng LangGraph

Đây là một hệ thống RAG (Retrieval Augmented Generation) được cấu trúc dưới dạng đồ thị, bao gồm:

1. Retrieval: Truy xuất tài liệu liên quan từ vector database
2. Grading: Đánh giá độ liên quan của tài liệu
3. Rewriting: Viết lại câu hỏi nếu cần
4. Generation: Tạo câu trả lời dựa trên tài liệu

Luồng xử lý:
- Người dùng đặt câu hỏi
- Agent quyết định có cần truy xuất tài liệu không
- Hệ thống đánh giá tài liệu và quyết định generate hoặc rewrite
- Cuối cùng tạo ra câu trả lời phù hợp
"""

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display
import pprint
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="../shared/.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

loader = PyPDFLoader(
    "../documents/2404.16130v1_Graph-RAG.pdf",
)
docs = loader.load()
docs[0]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_graph_rag",
    "Search and return information about Graph-RAG.",
)

tools = [retriever_tool]

class AgentState(TypedDict):
    """Lưu trữ các tin nhắn trong quá trình tương tác"""
    messages: Annotated[Sequence[BaseMessage], add_messages]

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Đánh giá mức độ liên quan của tài liệu được truy xuất với câu hỏi của người dùng.
    
    Args:
        state: Trạng thái hiện tại chứa lịch sử tin nhắn
        
    Returns:
        "generate": Nếu tài liệu liên quan đến câu hỏi
        "rewrite": Nếu tài liệu không liên quan và cần viết lại câu hỏi
    """
    print("---CHECK RELEVANCE---")

    # Mô hình dữ liệu
    class grade(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)

    # LLM với công cụ và xác thực
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"

def agent(state):
    """
    Agent chính xử lý việc tương tác với người dùng và quyết định các bước tiếp theo.
    
    Args:
        state: Trạng thái hiện tại chứa lịch sử tin nhắn
        
    Returns:
        dict: Trạng thái mới với phản hồi của agent
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

def rewrite(state):
    """
    Viết lại câu hỏi để tạo ra một câu hỏi tốt hơn khi tài liệu không liên quan.
    
    Args:
        state: Trạng thái hiện tại chứa lịch sử tin nhắn và câu hỏi gốc
        
    Returns:
        dict: Trạng thái mới với câu hỏi đã được viết lại
    """
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}

def generate(state):
    """
    Tạo câu trả lời dựa trên tài liệu liên quan đã được tìm thấy.
    
    Args:
        state: Trạng thái hiện tại chứa lịch sử tin nhắn và tài liệu
        
    Returns:
        dict: Trạng thái mới với câu trả lời được tạo ra
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

print("*" * 20 + "Prompt[rlm/rag-prompt]" + "*" * 20)
prompt = hub.pull("rlm/rag-prompt").pretty_print()  # Show what the prompt looks like

# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node(
    "generate", generate
)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile
graph = workflow.compile()

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

inputs = {
    "messages": [
        ("user", "What is Graph-RAG?"),
    ]
}
for output in graph.stream(inputs):
    for key, value in output.items():
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint("---")
        pprint.pprint(value, indent=2, width=80, depth=None)
    pprint.pprint("\n---\n")