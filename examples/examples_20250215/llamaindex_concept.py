from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, load_index_from_storage, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# embed_model = HuggingFaceEmbedding(
#     model_name="BAAI/bge-small-en-v1.5",
#     cache_folder="C:\\Users\\spano\\.cache\\huggingface\\hub")
# Settings.embed_model = embed_model

persist_dir = "./index_persist"
# 数据向量化，存入向量数据库
documents = SimpleDirectoryReader(input_files=["data/SQL题库.pdf"]).load_data()
# index = VectorStoreIndex.from_documents(documents, show_progress=True)
# index.storage_context.persist(persist_dir="./index_persist")
print(len(documents))
print(documents)
# print(index)

# index = load_index_from_storage(
#             StorageContext.from_defaults(persist_dir=persist_dir)
#     )
# 直接从向量库中查询数据，与大模型没有交互
# r = index.as_retriever(similarity_top_k=1)
# data = r.retrieve("第50题sql的答案是什么？")
# print(data)
# q = index.as_query_engine()
# data = q.query("第50题sql的答案是什么？")
# print(data)

