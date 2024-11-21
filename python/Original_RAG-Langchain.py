import bs4
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv(dotenv_path="../shared/.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

### Data ###
documents = [
    # France
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
    
    # USA
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
    
    # UK
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
    
    # Germany
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
    
    # Italy
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
    
    # South Korea
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
    
    # Japan
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
    
    # China
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
    
    # Vietnam
    "The capital of Vietnam is Hanoi.",
    "Hoan Kiem Lake is located in Hanoi.",
    "Vietnam is known for its pho and coffee.",
    "Ha Long Bay in Vietnam is a UNESCO World Heritage Site.",
    "Vietnam shares a border with China and Laos.",
    "Many Vietnamese people immigrated to the USA after the Vietnam War.",
    "The French colonial period influenced Vietnamese architecture and cuisine.",
    "Vietnam and South Korea have strong cultural ties through entertainment and trade.",
    "The Mekong Delta is an important agricultural region in Vietnam.",
    "Vietnamese students often study abroad in countries like Japan and Australia."
]

def get_retriever(documents: list):
    """
    Chuyển đổi danh sách văn bản thành hệ thống tìm kiếm thông minh
    
    Args:
        documents (list): Danh sách các đoạn văn bản cần xử lý
        
    Returns:
        tuple: Gồm 2 thành phần:
            - retriever: Công cụ tìm kiếm thông minh
            - vectorstore: Kho lưu trữ văn bản đã được chuyển đổi sang dạng vector
            
    Giải thích chi tiết:
    1. Chuyển mỗi đoạn văn bản thành đối tượng Document
    2. Chia nhỏ văn bản thành các đoạn dễ xử lý (1000 ký tự/đoạn)
    3. Chuyển văn bản thành vector số học để máy tính hiểu được
    4. Tạo công cụ tìm kiếm với khả năng trả về 2 kết quả gần nhất
    """
    try:
        # Tạo vector store và retriever
        docs = [Document(page_content=doc) for doc in documents]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        vectorstore = InMemoryVectorStore.from_documents(
            documents=splits, 
            embedding=OpenAIEmbeddings()
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        return retriever, vectorstore

    except Exception as e:
        print(f"Lỗi khi khởi tạo retriever: {str(e)}")
        default_doc = [
            Document(
                page_content="Có lỗi xảy ra khi tạo retriever. Vui lòng thử lại sau.",
                metadata={"source": "error"}
            )
        ]
        vectorstore = InMemoryVectorStore.from_documents(
            documents=default_doc, 
            embedding=OpenAIEmbeddings()
        )
        return vectorstore.as_retriever(), vectorstore

def search(query: str, llm: ChatOpenAI, retriever, memory: MemorySaver):
    """
    Thực hiện tìm kiếm thông minh và hiển thị quá trình suy luận của AI
    
    Args:
        query (str): Câu hỏi người dùng muốn tìm câu trả lời
        llm (ChatOpenAI): Mô hình AI để xử lý ngôn ngữ tự nhiên
        retriever: Công cụ tìm kiếm thông minh đã được tạo
        memory (MemorySaver): Bộ nhớ để lưu lại quá trình suy luận
        
    Returns:
        list: Danh sách các tin nhắn từ AI, bao gồm:
            - Các bước suy luận
            - Câu trả lời cuối cùng
            
    Quy trình hoạt động:
    1. Tạo công cụ tìm kiếm với khả năng tìm thông tin về các quốc gia
    2. Khởi tạo AI agent có khả năng suy luận
    3. Hiển thị từng bước suy luận của AI
    4. Trả về toàn bộ quá trình suy luận và kết quả
    """
    tool = create_retriever_tool(
        retriever,
        "find",
        "Search and return information about countries, including: Basic information, landmarks, culture, and international relations."
    )
    
    agent_executor = create_react_agent(llm, [tool], checkpointer=memory)
    config = {"configurable": {"thread_id": "abc123"}}
    
    try:
        print("\n🤔 Đang xử lý câu hỏi:", query)
        print("\n🔄 Quá trình suy luận của AI:")
        print("--------------------------------")
        
        results = []
        for event in agent_executor.stream(
            {"messages": [HumanMessage(content=query)]},
            config=config,
            stream_mode="values",
        ):
            # Hiển thị từng bước suy luận
            event["messages"][-1].pretty_print()
            print("--------------------------------")
            results.append(event["messages"][-1])
            
        return results
            
    except Exception as e:
        print(f"❌ Lỗi khi thực hiện tìm kiếm: {str(e)}")
        return [HumanMessage(content="Có lỗi xảy ra khi tìm kiếm. Vui lòng thử lại sau.")]

# Khởi tạo các thành phần
memory = MemorySaver()
llm = ChatOpenAI(model="gpt-4o", temperature=0)
retriever, vectorstore = get_retriever(documents)

# Example usage
if __name__ == "__main__":
    query = "I am learning about Vietnamese culture."
    results = search(query, llm, retriever, memory)
    # Chỉ in ra nội dung của message cuối cùng
    print('\n📝 Answer:')
    print(results[-1].content)

"""
Các bước chính của chương trình:
1. Tạo bộ nhớ để lưu quá trình suy luận
2. Khởi tạo mô hình AI GPT-4 với độ sáng tạo = 0 (để có câu trả lời nhất quán)
3. Tạo hệ thống tìm kiếm từ dữ liệu có sẵn
4. Chạy thử với câu hỏi về mối quan hệ văn hóa Pháp-Việt
"""