"""
使用最原始的方式，使用 openai 的 api 进行调用，不使用封装的类
需要安装 openai 包，并且需要设置 OPENAI_API_KEY 和 OPENAI_API_BASE 环境变量
相较于直接使用 openai 包，可以使用封装的类，例如 llama_index 的 chat_engine，可以更方便的使用
也可以使用 langchain 的 chat_engine，也可以更方便的使用

PS： 这两个封装的工具，在其他py文件中尝试
"""


import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_API_BASE"))

response = client.chat.completions.create(
    model="deepseek-chat",
    max_tokens=1024,
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "who are you?"},
    ],
    stream=False
)
print(response.choices[0].message.content)