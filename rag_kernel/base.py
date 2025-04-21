from abc import abstractmethod
from typing import List

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.storage.storage_context import DEFAULT_PERSIST_DIR, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore

import utils.settings as settings

class BaseRAG(object):
    def __init__(self, files: List[str]):
        self.files = files

    @abstractmethod
    async def load_file(self):
        """
        加载数据
        """

    async def create_local_rag_index(self, persist_dir: str = DEFAULT_PERSIST_DIR):
        """
        创建本地索引，该函数是数据库嵌入的重点优化模块
        数据向量存储在本地json文件中
        :return:
        """
        # 加载数据文件
        documents = await self.load_file()
        # 创建文件分割器
        node_splitter = SentenceSplitter.from_defaults(separator=".", chunk_size=512)
        # 利用分割器对文件进行分割，创建块
        nodes = node_splitter.get_nodes_from_documents(documents=documents)
        # 把分割好的块使用嵌入模型写入到向量数据库中，返回索引对象
        index = VectorStoreIndex(nodes, show_progress=True)
        # index = VectorStoreIndex.from_vector_store(nodes, show_progress=True)
        # 使用索引对象进行持久化操作
        index.storage_context.persist(persist_dir=persist_dir)

        # 返回索引对象
        return index

    async def create_remote_rag_index(self, collection_name="default"):
        # 加载文件
        documents = await self.load_file()
        # 创建分割器
        node_splitter = SentenceSplitter.from_defaults(chunk_size=384)
        # 使用分割器对文件进行分割，创建块
        nodes = node_splitter.get_nodes_from_documents(documents=documents)
        # 使用milvus向量库创建索引
        vector_store = MilvusVectorStore(
            uri=settings.configuration.milvus_uri,
            collection_name=collection_name,
            dim=settings.configuration.embedding_model_dimension,
            overwrite=True,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)

        return index

    @staticmethod
    def get_remote_rag_index(collection_name="default"):
        vector_store = MilvusVectorStore(
            uri=settings.configuration.milvus_uri,
            collection_name=collection_name,
            dimension=settings.configuration.embedding_model_dimension,
            overwrite=False,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

        return index
