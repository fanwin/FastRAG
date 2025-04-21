import os

from pydantic import BaseModel, Field


class Configuration(BaseModel):
    embedding_model_dimension: int = Field(
        default=384,
        description="The dimension of the embedding model.",
    )
    local_deepseek_model_name_1dot5b: str = Field(
        default=os.getenv("LOCAL_MODEL_NAME"),
        description="The name of the local model to use.",
    )
    local_deepseek_model_name_7b: str = Field(
        default=os.getenv("LOCAL_MODEL_NAME2"),
        description="The name of the local model to use.",
    )
    local_deepseek_api_key: str = Field(
        default=os.getenv("LOCAL_OPENAI_API_KEY"),
        description="The api key of the local model to use.",
    )
    local_deepseek_api_base: str = Field(
        default=os.getenv("LOCAL_OPENAI_API_BASE"),
        description="The api base of the local model to use.",
    )
    remote_deepseek_model_name: str = Field(
        default=os.getenv("MODEL_NAME"),
        description="The name of the deepseek remote model to use.",
    )
    remote_deepseek_api_key: str = Field(
        default=os.getenv("OPENAI_API_KEY"),
        description="The api key of the deepseek remote model to use.",
    )
    remote_deepseek_api_base: str = Field(
        default=os.getenv("OPENAI_API_BASE"),
        description="The api base of the deepseek remote model to use.",
    )
    local_embedding_model_name: str = Field(
        default=os.getenv("LOCAL_EMBEDDING_MODEL_NAME"),
        description="The name of the local embedding model to use.",
    )
    # local_embedding_model_dir: str = Field(
    #     default=os.getenv("LOCAL_EMBEDDING_MODEL_DIR"),
    #     description="The directory of the local embedding model to use.",
    # )
    remote_moonshot_model_name: str = Field(
        default=os.getenv("MOONSHOT_MODEL_NAME"),
        description="The name of the remote moonshot model to use.",
    )
    remote_moonshot_api_key: str = Field(
        default=os.getenv("MOONSHOT_API_KEY"),
        description="The api key of the remote moonshot model to use.",
    )
    remote_moonshot_api_base: str = Field(
        default=os.getenv("MOONSHOT_API_BASE"),
        description="The api base of the remote moonshot model to use.",
    )
    milvus_uri: str = Field(
        default=os.getenv("MILVUS_URI"),
        description="The uri of the milvus to use.",
    )

