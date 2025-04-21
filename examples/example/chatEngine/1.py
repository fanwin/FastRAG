import os

from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.openai.utils import ALL_AVAILABLE_MODELS, CHAT_MODELS
from dotenv import load_dotenv
from typing import Dict

# 创建一个模型会话引擎
load_dotenv()
Settings.llm = OpenAI(
    model=os.environ.get("MODEL_NAME"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    api_base=os.environ.get("OPENAI_API_BASE"),
)
USE_MODEL: Dict[str, int] = {
    "deepseek-chat": 100000,
}
ALL_AVAILABLE_MODELS.update(USE_MODEL)
CHAT_MODELS.update(USE_MODEL)
chat_engine = SimpleChatEngine.from_defaults()
# 开始会话
chat_engine.streaming_chat_repl()