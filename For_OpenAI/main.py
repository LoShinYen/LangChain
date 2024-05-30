from config import Config
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import os

# 配置參數
directory_path = Config.DATA_PATH
openai_api_key = Config.OPENAI_API_KEY
project_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(project_dir, "openai_chroma_storage")

def _split_data(directory_path):
    all_texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            
            # 使用 PyPDFLoader 讀取 PDF 文件
            loader = PyPDFLoader(file_path)
            
            # 設置文本分割器，將文本分割成多個 chunk
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            texts = loader.load_and_split(splitter)
            
            # 添加到總文本列表中
            all_texts.extend(texts)
    return all_texts

def main():
    # 建立本地向量存儲
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # 檢查是否存在持久化向量存儲
    if os.path.exists(persist_directory):
        vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    else:
        # 分割數據
        all_texts = _split_data(directory_path)
        vectorstore = Chroma.from_documents(all_texts, embeddings, persist_directory=persist_directory)
        vectorstore.persist()
    
    # 設置對話檢索鏈
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0, openai_api_key=openai_api_key), vectorstore.as_retriever())
    
    # 啟動對話系統
    chat_history = []
    while True:
        query = input('\nQ: ') 
        if not query:
            break
        result = qa({"question": query + ' (用繁體中文回答)', "chat_history": chat_history})
        print('A:', result['answer'])
        chat_history.append((query, result['answer']))

if __name__ == "__main__":
    main()
