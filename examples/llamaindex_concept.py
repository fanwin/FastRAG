import pprint

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, load_index_from_storage, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

embed_model = HuggingFaceEmbedding(
    # model_name="BAAI/bge-small-en-v1.5",
    model_name=r"D:\owin\AI\embed-model\bge-small-en\models--BAAI--bge-small-en-v1.5\snapshots\5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    cache_folder=r"D:\owin\AI\embed-model\bge-small-en")
Settings.embed_model = embed_model

persist_dir = "./index_persist"
# 数据向量化，存入向量数据库
# 一、分块后索引
documents = SimpleDirectoryReader(input_files=["data/SQL题库.pdf"]).load_data()
node_splitter = SentenceSplitter.from_defaults(separator="。", chunk_size=512)
nodes = node_splitter.get_nodes_from_documents(documents=documents)
index = VectorStoreIndex(nodes=nodes, show_progress=True)
index.storage_context.persist(persist_dir="./index_persist")

# 二、不分块，使用默认方式【但是其内部实现逻辑也是分块的，使用chunk 1024, overlap 200 个token】
# documents = SimpleDirectoryReader(input_files=["data/SQL题库.pdf"]).load_data()
# index = VectorStoreIndex.from_documents(documents=documents, show_progress=True)
# index.storage_context.persist(persist_dir="./index_persist")

# print(len(documents))
# pprint.pprint(documents)
# print(index)

# 数据已经持久化到向量数据库，直接加载索引
index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir=persist_dir)
)

# index 下有两种检索方式，一种是直接从向量库中查询数据，一种是从向量库中查询数据后，再与大模型进行交互
# 第一种检索方式 index.as_retriever() 获取检索器，然后调用检索器的retrieve方法，这里直接和向量数据库交互，【没有用到大模型】
# 第二种检索方式 index.as_query_engine() 获取查询引擎，然后调用查询引擎的query方法，这里需要和大模型交互，大模型会根据检索结果返回答案

# 第一种检索方式

# f = index.as_retriever(similarity_top_k=3)
# data = f.retrieve("第50题sql的答案是什么？")
# print(data)

# 第二种检索方式
# 逻辑中直接调用了 index.as_retriever 召回数据，【然后这些数据和用户的prompt信息一起喂给大模型】，大模型返回答案，【此时用到了大模型】
# 如果召回的数据错误，那么大模型会返回错误答案，用户会认为大模型有问题，用户会向大模型反馈错误答案，大模型会根据用户反馈的信息进行修正

# from llama_index.llms.deepseek import DeepSeek as DeepSeekOpenAI
# Settings.llm = DeepSeekOpenAI(
#         model="deepseek-chat",
#         api_base="https://api.deepseek.com/v1",
#         api_key="sk-4f1094cafbd541fbac3bf7fd5ceade48",
#     )
# q = index.as_query_engine(similarity_top_k=3)
# data = q.query("第50题sql的答案是什么？")
# print(data)

