from openai import OpenAI
from config import Config
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings.base import Embeddings
import os

directory_path = Config.DATA_PATH
llm_host = Config.LLM_HOST
llm_api_key = Config.LLM_API_KEY
llm_model = Config.LLM_MODEL
project_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(project_dir, "location_chroma_storage")

# 配置 LM Studio 客戶端
client = OpenAI(base_url=llm_host, api_key=llm_api_key)

def get_embedding(text, model=llm_model):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# 自定義嵌入類
class LMStudioEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [get_embedding(text) for text in texts]

    def embed_query(self, text):
        return get_embedding(text)

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
    # 初始化自定義嵌入類
    embeddings = LMStudioEmbeddings()
    
    # 檢查是否存在持久化向量存儲
    if os.path.exists(persist_directory):
        vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    else:
        # 分割數據
        all_texts = _split_data(directory_path)
        vectorstore = Chroma.from_documents(all_texts, embedding=embeddings, persist_directory=persist_directory)
        vectorstore.persist()
    
    # 設置檢索鏈
    retriever = vectorstore.as_retriever()

    # 初始化 ChatOpenAI 使用 LM Studio 作為 LLM
    lm_studio_model = ChatOpenAI(temperature=0, base_url=llm_host, api_key=llm_api_key)

    # 構建提示模板
    prompt_template = PromptTemplate(input_variables=["question"], template="Q: {question}\nA: ")

    # 初始化 LLMChain
    llm_chain = LLMChain(llm=lm_studio_model, prompt=prompt_template)

    # 設置對話檢索鏈
    qa_retrieval_chain = ConversationalRetrievalChain.from_llm(llm=lm_studio_model, retriever=retriever)

    # 啟動對話系統
    chat_history = []
    while True:
        query = input('\nQ: ')
        if not query:
            break

        # 使用檢索鏈進行問答
        result = qa_retrieval_chain({"question": query + ' (用繁體中文回答)', "chat_history": chat_history})
        answer = result.get('answer')

        print('A:', answer)
        chat_history.append((query, answer))

if __name__ == "__main__":
    main()
