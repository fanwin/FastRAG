import os
from typing import Dict

from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.llms.openai.utils import ALL_AVAILABLE_MODELS, CHAT_MODELS
from dotenv import load_dotenv
load_dotenv()

MY_MODE: Dict[str, int] = {
    "deepseek-chat": 100000,
}
ALL_AVAILABLE_MODELS.update(MY_MODE)
CHAT_MODELS.update(MY_MODE)
Settings.llm = LlamaIndexOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_API_BASE"),
)
# 使用llama_index创建一个聊天引擎对象
chat_engine = SimpleChatEngine.from_defaults()
# 使用引擎对象开启一个会话
chat_engine.streaming_chat_repl()