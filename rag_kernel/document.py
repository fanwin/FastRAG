from datetime import datetime
from typing import List
import os

from llama_index.core import SimpleDirectoryReader, Document

from rag_kernel import tools
from rag_kernel.base import BaseRAG


class DocumentRAGHandle(BaseRAG):
    async def load_file(self):
        docs = []
        for file in self.files:
            # 对图片及文档通过 Moonshot大 模型进行OCR识别
            # TODO 改造成离线识别，使用本地的OCR工具
            contents = tools.extract_text_from_llm(file)
            temp_file = datetime.now().strftime("%Y%m%d%H%M%S") + ".txt"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(contents)

            documents = SimpleDirectoryReader(input_files=[temp_file]).load_data()
            doc = Document(text="\n\n".join([d.text for d in documents]), metadata={"source": file})
            docs.append(doc)
            os.remove(temp_file)
        return docs