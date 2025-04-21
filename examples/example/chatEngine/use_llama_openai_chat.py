import os
from typing import Dict

from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.settings import Settings
from dotenv import load_dotenv
from llama_index.llms.openai.utils import ALL_AVAILABLE_MODELS, CHAT_MODELS

"""
1. 实现简单的chat；
2. 使用本地大模型例如deepseek，或者在线大模型deepseek-chat
3. 创建模型对象，使用llama_index 下的 OpenAI 类
4. 实现stream或者非 stream 的基于大模型的聊天
"""

"""
总结：大模型的框架（例如这里用到的 llama_index）不一定支持用到的大模型（例如这里的deepseek）
     要学会去改造；
     例如在 llama_index 0.12.23版本下， 如果不去更新 from llama_index.llms.openai.utils import ALL_AVAILABLE_MODELS, CHAT_MODELS
     则会报 deepseek-chat 不在有效的大模型范围内，改造后就可以执行；
     之所以可以改造，是因为要使用的大模型大多是兼容OpenAI接口的，如果不兼容就麻烦一些，查一下官网描述
"""

load_dotenv()

UPDATE_MODEL: Dict[str, int] = {
    "deepseek-chat": 200000,
    "deepseek-r1:1.5b": 200000,
    "deepseek-r1:7b": 200000,
}
ALL_AVAILABLE_MODELS.update(UPDATE_MODEL)
CHAT_MODELS.update(UPDATE_MODEL)

#
# 使用在线模型，不使用本地大模型
#
# Settings.llm = LlamaOpenAI(
#     model=os.environ.get("MODEL_NAME"),
#     api_key=os.environ.get("OPENAI_API_KEY"),
#     api_base=os.environ.get("OPENAI_API_BASE"))

# 使用本地化模型，不使用在线模型
Settings.llm = LlamaOpenAI(
    model=os.environ.get("LOCAL_MODEL_NAME"),
    api_key=os.environ.get("LOCAL_OPENAI_API_KEY"),
    api_base=os.environ.get("LOCAL_OPENAI_API_BASE"))

chat_engine = SimpleChatEngine.from_defaults()
# chat_engine.chat_repl()
# chat_engine.streaming_chat_repl()       # 使用流式输出


# 非流式输出
message = input("Human: ")
response = chat_engine.stream_chat(message)
print("Assistant: ", end="", flush=True)
response.print_response_stream()
print("\n")
