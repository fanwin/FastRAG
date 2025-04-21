import os
from dotenv import load_dotenv

# from llama_index.core.indices.struct_store import NLSQLTableQueryEngine
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.llms.deepseek import DeepSeek as DeepSeekOpenAI
from llama_index.core import SQLDatabase

from IPython.display import Markdown, display

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    insert,
)

load_dotenv()
def create_mysql_engine():
    con_mysql_engine = create_engine(
        "mysql+pymysql://root:hello123@localhost:3306/text2sql_db"
    )

    return con_mysql_engine

def create_sql_table(table_name: str = None):
    conn_mysql_engine = create_mysql_engine()
    metadata_obj = MetaData()
    # create city SQL table
    table_name = "city_stats" if not table_name else table_name
    city_stats_table = Table(
        table_name,
        metadata_obj,
        Column("city_name", String(16), primary_key=True),
        Column("population", Integer),
        Column("country", String(16), nullable=False),
    )
    metadata_obj.create_all(conn_mysql_engine)

def add_data_to_sql_table(table_name: str = None):
    sql_database = SQLDatabase(
        create_mysql_engine(),
        include_tables= [table_name],
    )
def llama_index_llm():
    llm = DeepSeekOpenAI(
        model=os.getenv("MODEL_NAME"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_API_BASE"),
        # local
        # model=os.getenv("LOCAL_MODEL_NAME"),
        # api_key=os.getenv("LOCAL_OPENAI_API_KEY"),
        # api_base=os.getenv("LOCAL_OPENAI_API_BASE"),
    )

    return llm

def query_use_nl(query_str: str = None):
    """
    使用自然语言查找数据库信息
    :param query_str:
    :return:
    """
    sql_database = SQLDatabase(
        create_mysql_engine(),
        include_tables= ["city_stats", "gdp_info"],
    )

    nl_query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["city_stats", "gdp_info"],
        llm=llama_index_llm()
    )
    query_str = "Which city has the highest population?" if not query_str else query_str
    response = nl_query_engine.query(query_str)
    display(Markdown(f"{response.response}"))
    print(response.response)

if __name__ == '__main__':
    # nl_query_str = "人口最少的城市是哪个？对这个城市进行描述"
    # nl_query_str = ("这里有哪些国家的哪些城市，用列表输出后；用python语言生成一个脚本，"
    #                 "脚本的逻辑是用饼状图来展示人口的分布，鼠标移上去可以查看城市名称和人口数量，表示城市的颜色区域弹跳出来")
    nl_query_str = ("GPD最高和最低的城市分别是哪个？")
    query_use_nl(query_str=nl_query_str)