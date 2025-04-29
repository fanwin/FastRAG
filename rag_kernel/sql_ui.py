"""
数据库RAG对话，单独提取出来，简化主 ui.py 文件的逻辑等级
"""
import chainlit as cl
from .database import SQLiteDatabase, MySQLDatabase

sql_rag = SQLiteDatabase()

async def is_sql_valid(sql_query):
    return sql_rag.is_sql_valid(sql_query)

async def train():
    await sql_rag.train_init_data()

@cl.step(language="sql", name="SQL生成智能体助手", show_input="text")
async def generate_sql(human_query: str, **kwargs)->str:
    current_step = cl.context.current_step
    current_step.input = human_query
    sql = sql_rag.generate_sql(human_query, allow_llm_to_see_data=True, **kwargs)
    current_step.output = sql

    return sql

@cl.step(name="SQL执行智能体助手", show_input="sql")
async def execute_query(sql: str):
    current_step = cl.context.current_step
    current_step.input = sql
    data_frame = sql_rag.run_sql(sql)
    current_output = data_frame.to_markdown()

    return data_frame

@cl.step(name="智能体分析助手", language="python")
async def plot(human_query, sql, data_frame):
    current_step = cl.context.current_step
    plot_code = sql_rag.generate_plotly_code(question=human_query, sql=sql, df_metadata=data_frame)
    figure = sql_rag.get_plotly_figure(plotly_code=plot_code, df=data_frame)
    current_step.output = plot_code

    return figure