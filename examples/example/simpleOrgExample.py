import os

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from examples.example.llms import deepseek_llm

load_dotenv()
Settings.openai_api_key = os.getenv("OPENAI_API_KEY")
Settings.openai_api_base = os.getenv("OPENAI_API_BASE")
Settings.llm = deepseek_llm()
# Settings.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
#                       base_url=os.getenv("OPENAI_API_BASE"))
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    cache_folder="D:\\owin\\AI\\embed-model\\bge-small-en",
)
# 创建一个简单的知识库
# 加载数据
documents = SimpleDirectoryReader(input_dir="D:\\owin\\doit\\AIAgent\\RAGDemo\\example\\data").load_data()

# 构造向量存储索引
index = VectorStoreIndex.from_documents(documents=documents)

# 构造查询引擎
query_engine = index.as_query_engine()

# 对查询引擎进行查询
responses = query_engine.query("人口最多的城市是哪个？")
print(responses)
