import torch
from transformers import DPRContextEncoder, DPRQuestionEncoder, DPRReader, T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np
import transformers
import os
import warnings
from transformers import utils
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.logging.set_verbosity_error()

# Sử dụng cached model nếu có
utils.use_cached_models = True

# Hoặc set offline mode
utils.offline_mode = True

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

def initialize_models():
    """
    Khởi tạo các mô hình cần thiết cho RAG
    
    Returns:
        tuple: Gồm các thành phần:
            - question_encoder: Mô hình mã hóa câu hỏi
            - context_encoder: Mô hình mã hóa ngữ cảnh
            - tokenizer: Tokenizer cho T5
            - index: FAISS index để tìm kiếm tương đồng
    """
    question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    index = faiss.IndexFlatIP(768)
    return question_encoder, context_encoder, tokenizer, index

def prepare_document_embeddings(documents, context_encoder, index):
    """
    Chuyển đổi documents thành vector embeddings và lưu vào FAISS index
    
    Args:
        documents (list): Danh sách các văn bản
        context_encoder: Mô hình mã hóa ngữ cảnh
        index: FAISS index
        
    Returns:
        numpy.ndarray: Ma trận embeddings của tất cả documents
    """
    doc_embeddings = []
    context_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    
    for doc in documents:
        inputs = context_tokenizer(doc, return_tensors="pt", truncation=True, padding=True, max_length=512)
        embeddings = context_encoder(**inputs).pooler_output.detach().numpy()
        doc_embeddings.append(embeddings[0])
        index.add(embeddings)
    
    return np.vstack(doc_embeddings)

def retrieve_documents(query, question_encoder, index, documents, k=3):
    """
    Tìm kiếm các documents liên quan nhất với câu hỏi
    
    Args:
        query (str): Câu hỏi của người dùng
        question_encoder: Mô hình mã hóa câu hỏi
        index: FAISS index
        documents (list): Danh sách các văn bản gốc
        k (int): Số lượng documents cần trả về
        
    Returns:
        list: Danh sách các documents liên quan nhất
    """
    # Sử dụng tokenizer riêng cho question_encoder
    question_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    inputs = question_tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Encode query thành vector
    query_embedding = question_encoder(**inputs).pooler_output.detach().numpy()
    
    # Lấy nhiều documents hơn ban đầu để có context tốt hơn
    k_search = k * 2
    distances, indices = index.search(query_embedding, k_search)
    
    # Sắp xếp documents theo độ liên quan (similarity score)
    retrieved_docs = []
    for i, dist in zip(indices[0], distances[0]):
        retrieved_docs.append((documents[i], dist))
    
    # Sắp xếp theo similarity score và lấy k documents có score cao nhất
    retrieved_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in retrieved_docs[:k]]

def generate_output(query, retrieved_docs, generator, tokenizer, method="sequence"):
    """
    Sinh câu trả lời dựa trên documents đã retrieve
    
    Args:
        query (str): Câu hỏi của người dùng
        retrieved_docs (list): Danh sách documents đã retrieve
        generator: Mô hình sinh câu trả lời
        tokenizer: Tokenizer cho mô hình sinh
        method (str): Phương pháp RAG ("sequence" hoặc "token")
        
    Returns:
        str: Câu trả lời được sinh ra
    """
    context = " ".join(retrieved_docs) if method == "sequence" else retrieved_docs[0]
    
    # Định dạng input cho T5
    prompt = f"question: {query} context: {context}"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = generator.generate(
        **inputs,
        max_length=128,
        min_length=20,
        num_beams=3,
        do_sample=False,  # Tắt sampling để có kết quả ổn định hơn
        temperature=1.0,
        early_stopping=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    """
    Hàm chính thực thi pipeline RAG
    """
    print("\n🤖 Initializing RAG pipeline...")
    
    question_encoder, context_encoder, tokenizer, index = initialize_models()
    
    print("📚 Preparing document embeddings...")
    doc_embeddings = prepare_document_embeddings(documents, context_encoder, index)
    
    # Khởi tạo T5 thay vì BART
    generator = T5ForConditionalGeneration.from_pretrained("t5-base")
    
    query = "I am learning about Vietnamese culture."
    print(f"\n❓ Question: {query}")
    
    retrieved_docs = retrieve_documents(query, question_encoder, index, documents, k=3)
    print("\n📄 Retrieved documents:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"{i}. {doc}")
    
    print("\n🔍 RAG-Sequence Response:")
    response_sequence = generate_output(query, retrieved_docs, generator, tokenizer, method="sequence")
    print(response_sequence)
    
    print("\n🔍 RAG-Token Response:")
    response_token = generate_output(query, retrieved_docs, generator, tokenizer, method="token")
    print(response_token)

if __name__ == "__main__":
    main()
