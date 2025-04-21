# llama_index 已经支持deepseek，不需要更新 ALL_VALIDABLE_MODELS 和 CHAT_MODELS
# 尝试做一下，需要安装 pip install llama-index-llm-deepseek
import os

from llama_index.core import Settings
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.deepseek import DeepSeek as DeepSeekOpenAI
from dotenv import load_dotenv
load_dotenv()

# Settings.llm = DeepSeekOpenAI(
#     model="deepseek-chat",
#     temperature=0.1,
#     max_tokens=1024,
#     api_key=os.getenv("OPENAI_API_KEY"),
#     api_base=os.getenv("OPENAI_API_BASE")
# )
Settings.llm = DeepSeekOpenAI(
    model=os.getenv("LOCAL_MODEL_NAME"),
    temperature=0.1,
    max_tokens=1024,
    api_key=os.getenv("LOCAL_OPENAI_API_KEY"),
    api_base=os.getenv("LOCAL_OPENAI_API_BASE")
)

chat_engine = SimpleChatEngine.from_defaults()
chat_engine.streaming_chat_repl()