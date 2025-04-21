import os

from llama_index.llms.openai import OpenAI


def deepseek_llm():
    client = OpenAI(model="deepseek-chat",
           api_base=os.environ.get("OPENAI_API_BASE"),
           api_key=os.environ.get("OPENAI_API_KEY"),
           temperature=0)

    return client
llm = deepseek_llm()
response = llm.chat(
    model="deepseek-chat",
    max_tokens=1024,
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "who are you?"},
    ],
    stream=False
)
print(response.choices[0].message.content)