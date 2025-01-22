import json
import os
import sys

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
# 将项目根目录添加到 sys.path 中
sys.path.insert(0, project_root)

import random
from langchain_community.chat_models import ChatZhipuAI
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, Field
from itext2kg_builder import iText2KG
from itext2kg_builder.documents_distiller.documents_distiller import DocumentsDistiller
import ollama
import config
from itext2kg_builder.graph_integration import GraphIntegrator
from langchain_core.runnables import Runnable


# 读取文件并整理格式
def read_and_clean_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # 去除所有换行符
            content_without_newlines = content.replace('\n', '').replace('\r', '')
            return content_without_newlines
    except FileNotFoundError:
        print(f"文件未找到：{file_path}")
        return None
    except IOError:
        print(f"读取文件时发生错误：{file_path}")
        return None


class OllamaClientAdapter(Runnable):
    def __init__(self, ollama_client, model_name):
        self.ollama_client = ollama_client
        self.model_name = model_name

    def invoke(self, input_text, config=None):
        response = self.ollama_client.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': str(input_text)}],
            stream=False
        )
        return response['message']['content']

class Character(BaseModel):
    name: str = Field(
        ..., description="人物名称"
    )
    description: str = Field(..., description="人物的介绍")

def build(base_path, doc_id):

    # client = ollama.Client(host=config.ollama_url)
    # llm = OllamaClientAdapter(client, config.llm_name)
    llm = ChatZhipuAI(model=config.zhipu_model_name, api_key=config.zhipu_api_key)

    ollama.pull("nomic-embed-text")
    embedding = OllamaEmbeddings(
        model="nomic-embed-text",
    )

    # 初始化DocumentDistiller
    document_distiller = DocumentsDistiller(llm_model = llm)

    # documents = [os.path.join(base_path, f) for f in os.listdir(base_path)]

    # 信息提取查询
    IE_query = '''
    # 指令：
    - 您将阅读一段文字，请用简洁和专业的文字来回答问题
    - 提取小说中所有的人物角色信息和关系，
    - 所有的名称和关系均使用中文
    - 不要基于假设或猜测提供信息，只使用文章内容。如果无法从中得到答案，则不回答该问题。
    '''

    # 使用定义的查询和输出数据结构提取文档
    distilled_doc, dic = document_distiller.distill(documents=base_path, IE_query=IE_query, output_data_structure=Character)

    # 用llm模型和嵌入模型初始化iText2KG
    itext2kg = iText2KG(llm_model = llm, embeddings_model = embedding)

    # 将提炼出来的文档格式化为语义部分
    semantic_blocks = [f"{key} - {value}".replace("{", "[").replace("}", "]")
                       for key, value in distilled_doc.items()]

    # 使用语义部分构建知识图谱   entity+relation
    return itext2kg.build_graph(sections=semantic_blocks, doc_id=doc_id, dic=dic)

def to_neo4j(ent, rel):
    URI = config.neo4j_local_ip
    USERNAME = config.neo4j_user
    PASSWORD = config.neo4j_password
    new_graph = {"nodes": kg_ent, "relationships": kg_rel}
    GraphIntegrator(uri=URI, username=USERNAME, password=PASSWORD).visualize_graph(json_graph=new_graph)
    print('Build successfully!')

path = 'datasets2/xiyou.txt'
doc_id = 'qwertyuiop'
st = read_and_clean_file(path)
print(st)
kg_ent, kg_rel = build(st, doc_id)
# to_neo4j(kg_ent, kg_rel)




