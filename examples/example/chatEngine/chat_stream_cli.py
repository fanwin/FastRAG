import os
from dotenv import load_dotenv

from openai import OpenAI


def local_deepseek_llm():
    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("LOCAL_OPENAI_API_KEY"),
        base_url=os.getenv("LOCAL_OPENAI_API_BASE")
    )

    return client
def remote_deepseek_llm():
    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )

    return client

def base_simple_chat(chat_mode: int):
    client = remote_deepseek_llm() if chat_mode == "remote" else local_deepseek_llm()
    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME") if chat_mode == 1 else os.getenv("LOCAL_MODEL_NAME"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )
    print(response.choices[0].message.content)

def chat_stream(chat_mode: int):
    """
    开启流式对话
    :param chat_mode: 0 使用本地部署的大模型，1 表示使用远端大模型api
    :return:
    """
    client = remote_deepseek_llm() if chat_mode == 1 else local_deepseek_llm()
    message_context = [{"role": "system", "content": "You are a helpful assistant."}]

    try:
        while True:
            # 用户输入
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            # 添加用户输入到消息上下文
            message_context.append({"role": "user", "content": user_input})

            # 调用模型接口，获取模型回复
            response = client.chat.completions.create(
                model=os.getenv("MODEL_NAME") if chat_mode == 1 else os.getenv("LOCAL_MODEL_NAME"),
                messages=message_context,
                temperature=0,
                max_tokens=4096,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=True
            )

            # 处理流式输出，每次迭代都会返回一个token
            print("Assisant: ", end="", flush=True)
            assistant_response = process_stream_response(response)

            # 维护对话历史，将模型回复添加到消息上下文
            # message_context.append({"role": "assistant", "content": assistant_response})
    except Exception as e:
        print("\n发生错误", {str(e)})
    except KeyboardInterrupt:
        print("\n用户已终止")

def process_stream_response(response):
    full_response = []
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response.append(content)
    print()  # 换行分隔回复
    return "".join(full_response)

if __name__ == '__main__':
    which_model = input("which model do you want to use? (local or remote): \n0 is local\n1 is remote\n")
    # base_simple_chat(which_model)
    chat_stream(which_model)
