import os
import random
from typing import List, Any
import chainlit as cl
import chainlit.data as cl_data
from chainlit.element import ElementBased
from llama_index.core import Settings
from llama_index.core.chat_engine.types import ChatMode, BaseChatEngine
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from rag_kernel.document import DocumentRAGHandle
from rag_kernel import sql_ui
from utils import settings
from persistent.minio_storage_client import MinioStorageClient
from persistent.postgresql_data_layer import PostGreSQLDataLayer

# storage_client = MinioStorageClient()
# cl_data._data_layer = PostGreSQLDataLayer(conninfo=settings.configuration.pg_connection_string, storage_provider=storage_client)

async def view_pdf(elements: List[ElementBased]):
    """
    查看PDF文件
    如果不是pdf文件，则任务是默认和大模型的普通对话，此时 files 长度为0
    """
    files = []
    contents = []
    for element in elements:
        if element.name.endswith(".pdf"):
            pdf = cl.Pdf(name=element.name, display="side", path=element.path)
            files.append(pdf)
            contents.append(element.name)
    if len(files) == 0:
        return
    await cl.Message(content=f"查看PDF文件：" + "，".join(contents), elements=files).send()

@cl.on_chat_start
async def start():
    chat_engine = None
    chat_profile = cl.user_session.get("chat_profile")
    if chat_profile == "普通对话" or chat_profile == "上传文件对话" or chat_profile == "数据库对话":
        memory = ChatMemoryBuffer.from_defaults(token_limit=2048)
        chat_engine = SimpleChatEngine.from_defaults(memory=memory)
    elif chat_profile == "知识库问答":
        remote_index = DocumentRAGHandle.get_remote_rag_index(collection_name=chat_profile)
        chat_engine = remote_index.as_chat_engine(chat_mode=ChatMode.CONTEXT, similarity_top_k=15)
    cl.user_session.set("chat_engine", chat_engine)

@cl.set_chat_profiles
async def my_profiles():
    # 切换聊天模式，例如 大模型对话、知识库问答等
    profiles = [
        cl.ChatProfile(
            name="普通对话",
            markdown_description="这是一个普通的大模型对话模式，和大模型直接对话",
            icon=f"/public/kbs/model.png"
        ),
        cl.ChatProfile(
            name="上传文件对话",
            markdown_description="也是普通对话的一种，上传文件后和大模型进行交互",
            icon=f"/public/kbs/db.jpg"
        ),
        cl.ChatProfile(
            name="数据库对话",
            markdown_description="用自然语言和数据库对话（text2SQL）",
            icon=f"/public/kbs/db.jpg"
        ),
        # cl.ChatProfile(
        #     name="知识库问答",
        #     markdown_description="知识库问答，利用text2sql技术",
        #     icon=f"/public/kbs/db.jpg"
        # ),
    ]
    # 知识库可以根据企业实际的业务场景来进行设置
    # 例如可以按app区分知识库，按特定场景分知识库等等【财务、法务、人事、运营......】
    # 数据存储在向量数据库或者图谱结构中，例如 Milvus、Pinecone、ChromaDB、Qdrant、FAISS等
    # define_profile = settings.milvus_client.list_collections()
    # for item in define_profile:
    #     profiles.append(
    #         cl.ChatProfile(
    #             name=item,
    #             markdown_description=f"{item} 知识库对话，",
    #             icon=f"/public/kbs/{random.randint(1,3)}.jpg"
    #         )
    #     )

    return profiles

# 装饰器on_message表示当用户发送消息时，会自动调用其装饰的异步函数，
# 并传入一个chainlit.Message的对象message，表示用户发送的消息
@cl.on_message
async def main(message: cl.Message):
    # 初始化一个空的消息助手对象，用于逐步构建助手的响应
    # author="Assistant" 表示消息的作者或发送者是助手，会显示在前端会话历史中
    msg = cl.Message(content="", author="Assistant")
    chat_mode = cl.user_session.get("chat_profile", "大模型对话")

    if chat_mode == "普通对话" or chat_mode == "上传文件对话":
        files = []
        await view_pdf(message.elements)
        for element in message.elements:
            if isinstance(element, cl.File) or isinstance(element, cl.Image):
                files.append(element.path)
        if len(files) > 0:  # 如果用户上传了文件，则创建rag_index
            rag = DocumentRAGHandle(files=files)
            index = await rag.create_local_rag_index()
            chat_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT, similarity_top_k=18)
            # re = index.as_retriever(similarity_top_k=3)
            # data = re.retrieve("第22题的答案是多少")
            # print(data)
            cl.user_session.set("chat_engine", chat_engine)
    elif chat_mode == "知识库对话":
        pass
    elif chat_mode == "数据库对话":      # text2SQL 技术
        # await sql_ui.train()
        sql = await sql_ui.generate_sql(message.content)
        is_valid = await sql_ui.is_sql_valid(sql)
        if is_valid:
            df = await sql_ui.execute_query(sql)
            await cl.Message(content=df.to_markdown(index=False), author="Assistant").send()
            fig = await sql_ui.plot(human_query=message.content, sql=sql, data_frame=df)
            elements = [cl.Plotly(name="chart", figure=fig, display="inline")]
            await cl.Message(content="生成的图表如下：", elements=elements, author="Assistant").send()
            return
    else:
        raise ValueError("不支持的聊天模式")
    # 获取聊天引擎
    chat_engine = cl.user_session.get("chat_engine")
    # 使用chainlit.make_async，把同步函数chat_engine.stream_chat转换为异步协程函数后，接收输入参数message.content，并返回一个生成器
    res = await cl.make_async(chat_engine.stream_chat)(message.content)
    # 开始处理流式响应，res.response_gen是一个生成器(逐步生成响应内容)，每次迭代都会返回一个token
    for token in res.response_gen:  # 遍历生成器，每次迭代都会返回一个token，通常是一个词语或词语片段
        # await 确保每个token都处理完毕发送到前端
        await msg.stream_token(token)  # 将token逐步添加到助手msg中（content+=token），实现流式响应输出的动态更新，达流式输出的效果

    # 如果当前对话是知识库对话，则显示数据来源
    await get_node_of_acknowledgement(chat_engine, msg, res)

    # 所有的token处理完成后，将助手msg详细发送到前端
    await msg.send()

async def get_node_of_acknowledgement(chat_engine: BaseChatEngine, msg: cl.Message, res: Any):
    # 如果当前对话是知识库对话，则显示数据来源，显示个数和 similarity_top_k 个数一致
    if not isinstance(chat_engine, SimpleChatEngine):
        source_names = []
        for idx, node_with_score in enumerate(res.source_nodes):
            node = node_with_score.node
            source_name = f"source_{idx}"
            source_names.append(source_name)
            msg.elements.append(
                cl.Text(content=node.get_text(),
                        name=source_name,
                        display="side")
            )
        await msg.stream_token(f"\n\n **数据来源**: {', '.join(source_names)}")
