# 12-Types-Of-RAG

    - Link post: https://www.turingpost.com/p/12-types-of-rag

# 12 Types of RAG - Part 1 - Original RAG

1. Tài liệu:
   - Trong thư mục "documents", file "2005.11401v4_Original-RAG.pdf"
   - Tạo file .env trong folder shared (ví dụ ở file .env.example)
2. Với python:
   - Tạo môi trường với anaconda qua lệnh: `conda create -n myenv python=3.11`
   - Sau đó active enviroment qua câu lệnh: `conda activate myenv`
   - Cài đặt các package qua lệnh: `pip install -r requirements.txt`
   - cd vào folder python: `cd python`
   - Chạy file: `python Original_RAG.py`: File code với 2 phương pháp của trong bài báo: (RAG-Sequence và RAG-Token)
   - Chạy file: `python Original_RAG-Langchain.py`: File code với Langchain python
3. Với Typescript:
   - cd vào folder typescript: `cd typescript`
   - Cài đặt môi trường qua lệnh: `npm install`
   - Chạy file: `tsx Original_RAG-Langchain.ts`: File code với LangchainJS

# Note (cài đặt package thủ công):

1. Python:

   - `pip install transformers faiss-cpu torch`
   - `pip install sentencepiece`
   - `pip install langchain langchain-community langchain-openai beautifulsoup4`
   - `pip install python-dotenv langgraph`

2. Typescript:
   - `npm install -g tsx`
   - `sudo npm install -g tsx`
   - `npm init -y`
   - `npm install --save langchain @langchain/openai langchain @langchain/langgraph @langchain/core @langchain/community dotenv`
