import os.path

from chainlit.utils import mount_chainlit
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse
load_dotenv()
from rag_kernel.document import DocumentRAGHandle
from utils.response import Response
app = FastAPI()

@app.post("/files/")
async def create_upload_files(files: list[UploadFile], collection_name: str = Form()):
    file_list = []
    for file in files:
        file_path = os.path.join("data", os.path.basename(file.filename))
        with open(file_path, "wb+") as f:
            f.write(await file.read())
            file_list.append(file_path)
    rag = DocumentRAGHandle(files=file_list)
    await rag.create_remote_rag_index(collection_name=collection_name)
    return Response.ok("index success")


@app.get("/")
async def main():
    content = """
        <body>
        <form action="/files/" enctype="multipart/form-data" method="post">
            <input name="files" type="file" multiple>
            <input name="collection_name" type="text">
            <input type="submit">
        </form>
        </body>
    """
    return HTMLResponse(content=content)

mount_chainlit(app=app, target="ui.py", path="/chainlit")
# 启动fastapi命令： uvicorn main:app --host 0.0.0.0 --port 80
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
