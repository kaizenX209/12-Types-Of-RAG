from typing import List, Dict, Any
import spacy
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from community import community_louvain
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import transformers
import warnings
from transformers import utils
from pypdf import PdfReader
import re

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.logging.set_verbosity_error()

# Sử dụng cached model nếu có
utils.use_cached_models = True

# Hoặc set offline mode
utils.offline_mode = True

load_dotenv(dotenv_path="../shared/.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

class GraphRAG:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm = ChatOpenAI()
        self.knowledge_graph = nx.Graph()

    def read_pdf(self, pdf_path: str) -> List[str]:
        """
        Đọc và xử lý file PDF thành list các đoạn văn bản
        """
        documents = []
        reader = PdfReader(pdf_path)
        
        for page in reader.pages:
            text = page.extract_text()
            
            # Làm sạch text
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            # Chia thành các đoạn dựa trên dấu chấm câu
            paragraphs = re.split(r'(?<=[.!?])\s+', text)
            documents.extend([p for p in paragraphs if len(p) > 50])  # Chỉ lấy đoạn có ý nghĩa
            
        return documents

    def process_source_to_chunks(self, documents: List[str]) -> List[str]:
        """
        2.1 Source Documents → Text Chunks
        Chia văn bản nguồn thành các đoạn nhỏ
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text("\n".join(documents))
        return chunks

    def chunks_to_elements(self, chunks: List[str]) -> List[Dict]:
        """
        2.2 Text Chunks → Element Instances
        Trích xuất các phần tử (thực thể, quan hệ, khái niệm) từ chunks
        """
        elements = []
        for chunk in chunks:
            doc = self.nlp(chunk)
            
            # Trích xuất thực thể
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            # Trích xuất noun chunks
            noun_chunks = [chunk.text for chunk in doc.noun_chunks]
            
            # Trích xuất quan hệ
            relations = []
            for token in doc:
                if token.dep_ in ['nsubj', 'dobj', 'pobj']:
                    relations.append({
                        'source': token.text,
                        'target': token.head.text,
                        'type': token.dep_
                    })
            
            elements.append({
                'text': chunk,
                'entities': entities,
                'noun_chunks': noun_chunks,
                'relations': relations,
                'embedding': self.encoder.encode(chunk)
            })
        
        return elements

    def elements_to_summaries(self, elements: List[Dict]) -> List[Dict]:
        """
        2.3 Element Instances → Element Summaries
        Tạo tóm tắt cho mỗi phần tử
        """
        summaries = []
        for element in elements:
            # Tạo prompt để tóm tắt
            prompt = f"""Summarize the following text and its key elements:
            Text: {element['text']}
            Entities: {element['entities']}
            Key concepts: {element['noun_chunks']}
            """
            
            # Thay predict() bằng invoke()
            summary = self.llm.invoke(prompt).content
            summaries.append({
                'original': element,
                'summary': summary,
                'embedding': self.encoder.encode(summary)
            })
        
        return summaries

    def summaries_to_communities(self, summaries: List[Dict]) -> Dict:
        """
        2.4 Element Summaries → Graph Communities
        Xây dựng đồ thị và phát hiện cộng đồng
        """
        # Xây dựng đồ thị từ các tóm tắt
        for summary in summaries:
            self.knowledge_graph.add_node(
                summary['summary'],
                original=summary['original'],
                embedding=summary['embedding']
            )
        
        # Thêm edges dựa trên similarity
        nodes = list(self.knowledge_graph.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                emb1 = self.knowledge_graph.nodes[nodes[i]]['embedding']
                emb2 = self.knowledge_graph.nodes[nodes[j]]['embedding']
                similarity = cosine_similarity([emb1], [emb2])[0][0]
                
                if similarity > 0.5:
                    self.knowledge_graph.add_edge(nodes[i], nodes[j], weight=similarity)
        
        # Phát hiện cộng đồng
        communities = community_louvain.best_partition(self.knowledge_graph)
        return communities

    def communities_to_summaries(self, communities: Dict) -> List[Dict]:
        """
        2.5 Graph Communities → Community Summaries
        Tạo tóm tắt cho mỗi cộng đồng
        """
        community_groups = {}
        for node, community_id in communities.items():
            if community_id not in community_groups:
                community_groups[community_id] = []
            community_groups[community_id].append(node)
        
        community_summaries = []
        for community_id, nodes in community_groups.items():
            community_content = "\n".join(nodes)
            prompt = f"""Create a comprehensive summary for this group of related information:
            Content: {community_content}
            """
            
            # Thay predict() bằng invoke()
            summary = self.llm.invoke(prompt).content
            community_summaries.append({
                'community_id': community_id,
                'nodes': nodes,
                'summary': summary
            })
        
        return community_summaries

    def generate_global_answer(self, query: str, community_summaries: List[Dict]) -> str:
        """
        2.6 Community Summaries → Community Answers → Global Answer
        Tạo câu trả lời cuối cùng từ các tóm tắt cộng đồng
        """
        query_embedding = self.encoder.encode(query)
        relevant_communities = []
        
        for comm_summary in community_summaries:
            summary_embedding = self.encoder.encode(comm_summary['summary'])
            similarity = cosine_similarity([query_embedding], [summary_embedding])[0][0]
            if similarity > 0.3:
                relevant_communities.append(comm_summary)
        
        community_answers = []
        for comm in relevant_communities:
            prompt = f"""Based on this community summary, answer the question:
            Summary: {comm['summary']}
            Question: {query}
            """
            # Thay predict() bằng invoke()
            answer = self.llm.invoke(prompt).content
            community_answers.append(answer)
        
        final_prompt = f"""Combine these community-specific answers into a coherent global answer:
        Question: {query}
        Community Answers: {' '.join(community_answers)}
        """
        
        # Thay predict() bằng invoke()
        global_answer = self.llm.invoke(final_prompt).content
        return global_answer

    

def main():
    # Khởi tạo GraphRAG
    rag = GraphRAG()
    
    # Sửa lại đường dẫn PDF, chỉ giữ một đường dẫn đúng
    pdf_path = "../documents/2404.16130v1_Graph-RAG.pdf"
    
    # Kiểm tra xem file có tồn tại không
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Không tìm thấy file PDF tại đường dẫn: {pdf_path}")
        
    print(f"Đang đọc PDF từ {pdf_path}")
    documents = rag.read_pdf(pdf_path)
    
    # Thực hiện từng bước trong pipeline
    chunks = rag.process_source_to_chunks(documents)
    elements = rag.chunks_to_elements(chunks)
    element_summaries = rag.elements_to_summaries(elements)
    communities = rag.summaries_to_communities(element_summaries)
    community_summaries = rag.communities_to_summaries(communities)
    
    # Xử lý câu hỏi
    query = "What is Graph-RAG?"
    answer = rag.generate_global_answer(query, community_summaries)
    
    print(f"Question: {query}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
