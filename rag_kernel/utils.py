from pathlib import Path

from openai import OpenAI

from utils.settings import remote_moonshot_llm

def extract_text_from_llm(file_path):
    client = OpenAI(
        api_key="sk-IrjHqno9H4w0LFhpac33IvzUBRceMapE6I76brB1WaDCPSLd",
        base_url="https://api.moonshot.cn/v1",
    )
    file_object = client.files.create(file=Path(file_path), purpose="file-extract")
    file_content = client.files.content(file_id=file_object.id).json()

    return file_content.get('content')
