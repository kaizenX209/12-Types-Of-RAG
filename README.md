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

# 12 Types of RAG - Part 2 - Graph RAG

1. Tài liệu:

   - Trong thư mục "documents", file "2404.16130v1_Graph-RAG.pdf"

2. Với python:

   - Active enviroment ở part 1 qua câu lệnh: `conda activate myenv`
   - Cài đặt package qua lệnh: `pip install -r requirements.txt`
   - cd vào folder python: `cd python`
   - Chạy file: `python Graph_RAG.py`: File code theo bài báo
   - Chạy file: `python Graph_RAG-Langgraph.py`: File code với Langgraph python

3. Với Typescript:
   - cd vào folder typescript: `cd typescript`
   - Cài đặt môi trường qua lệnh: `npm install`
   - Chạy file: `tsx Graph_RAG-Langgraph.ts`: File code với Langgraph Typescript

# Note (cài đặt package thủ công):

1. Python:

   - Part 1:

   * `pip install transformers faiss-cpu torch`
   * `pip install sentencepiece`
   * `pip install langchain langchain-community langchain-openai beautifulsoup4`
   * `pip install python-dotenv langgraph`

   - Part 2:

   * `pip uninstall numpy -y`
   * `pip install numpy==1.26.4`
   * `pip uninstall blis thinc pydantic langchain -y`
   * `pip install "pydantic==2.9.2" "blis==0.7.9" "thinc==8.2.5" "langchain==0.3.7" "langchain-community==0.3.7"`
   * `pip uninstall spacy -y`
   * `pip install spacy==3.7.4`
   * `pip install networkx sentence-transformers scikit-learn python-louvain pypdf`
   * `pip install ipython`
   * `pip install tiktoken langchainhub chromadb langchain-text-splitters`

2. Typescript:

   - Part 1:

   * `npm install -g tsx`
   * `sudo npm install -g tsx`
   * `npm init -y`
   * `npm install --save langchain @langchain/openai langchain @langchain/langgraph @langchain/core @langchain/community dotenv`

- Part 2:

  - `npm install cheerio zod zod-to-json-schema @langchain/textsplitters`
  - `npm i @langchain/community @langchain/core pdf-parse`
