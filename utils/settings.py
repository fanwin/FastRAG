from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.deepseek import DeepSeek as DeepSeekOpenAI
from openai import OpenAI
from pymilvus import model
from utils.config import Configuration

configuration = Configuration()

# milvus_client = MilvusClient(uri=configuration.milvus_uri)
def list_all_milvus_collections():
    """
    返回所有milvus向量库中的collection
    :return:
    """
    return milvus_client.list_collections()

def drop_milvus_collection(collection_name):
    """
    删除milvus向量库中的collection by name
    :param collection_name:
    :return:
    """
    milvus_client.drop_collection(collection_name)

def local_llama_index_deepseek_llm():
    return DeepSeekOpenAI(
        model=configuration.local_deepseek_model_name_1dot5b,
        api_base=configuration.local_deepseek_api_base,
        api_key=configuration.local_deepseek_api_key,
    )

def remote_llama_index_deepseek_llm():
    return DeepSeekOpenAI(
        model=configuration.remote_deepseek_model_name,
        api_base=configuration.remote_deepseek_api_base,
        api_key=configuration.remote_deepseek_api_key,
    )

def remote_moonshot_llm():
    return OpenAI(
        base_url=configuration.remote_moonshot_api_base,
        api_key=configuration.remote_moonshot_api_key
    )
def local_embedding_model():
    embedding_model = HuggingFaceEmbedding(
        model_name=configuration.local_embedding_model_name,
    )

    return embedding_model

def pymilvus_bge_small_embedding_function(**kwargs):
    return model.dense.SentenceTransformerEmbeddingFunction(
        model_name=configuration.local_embedding_model_name,    # 'BAAI/bge-small-en-v1.5'
        device='cpu',
    )

def openai_llm(**kwargs):
    return OpenAI(api_key=configuration.llm_api_key,
                  base_url=configuration.llm_api_base, **kwargs)

# 设置全局配置
# Settings.llm = local_llama_index_deepseek_llm()
Settings.llm = remote_llama_index_deepseek_llm()
Settings.embed_model = local_embedding_model()
