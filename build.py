from flask import Flask, request, jsonify
import os
import sys
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
# 将项目根目录添加到 sys.path 中
sys.path.insert(0, project_root)
import random
import json
from langchain_community.chat_models import ChatZhipuAI
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, Field
from itext2kg_builder import iText2KG
from itext2kg_builder.documents_distiller.documents_distiller import DocumentsDistiller
import ollama
import config
from itext2kg_builder.graph_integration import GraphIntegrator
from langchain_core.runnables import Runnable

app = Flask(__name__)

class OllamaClientAdapter(Runnable):
    def __init__(self, ollama_client, model_name):
        self.ollama_client = ollama_client
        self.model_name = model_name

    def invoke(self, input_text, config_=None):
        response = self.ollama_client.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': str(input_text)}],
            stream=False
        )
        return response['message']['content']


class Character(BaseModel):
    name: str = Field(..., description="人物名称")
    description: str = Field(..., description="人物的介绍")


def build(base_path, doc_id):
    # client = ollama.Client(host=config.ollama_url)
    # llm = OllamaClientAdapter(client, config.llm_name)
    llm = ChatZhipuAI(model=config.zhipu_model_name,
                      api_key=config.zhipu_api_key)
    ollama.pull("nomic-embed-text")
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    document_distiller = DocumentsDistiller(llm_model=llm)

    # documents = [os.path.join(base_path, f) for f in os.listdir(base_path)]

    IE_query = '''
    # 指令：
    - 您将阅读一段文字，
    - 提取小说中所有的人物角色信息和关系，
    - 所有内容全部使用中文回答
    '''

    text = base_path.replace('\n', '').replace('\r', '')
    distilled_doc, dic = document_distiller.distill(documents=text, IE_query=IE_query,
                                                    output_data_structure=Character)
    itext2kg = iText2KG(llm_model=llm, embeddings_model=embedding)
    semantic_blocks = [f"{key} - {value}".replace("{", "[").replace("}", "]") for key, value in distilled_doc.items()]
    return itext2kg.build_graph(sections=semantic_blocks, doc_id=doc_id, dic=dic)


def to_neo4j(entities, relationships):
    URI = config.neo4j_local_ip
    USERNAME = config.neo4j_user
    PASSWORD = config.neo4j_password
    new_graph = {"nodes": entities, "relationships": relationships}
    GraphIntegrator(uri=URI, username=USERNAME, password=PASSWORD).visualize_graph(json_graph=new_graph)
    return {"message": "Knowledge Graph visualized successfully"}


@app.route('/build_kg', methods=['POST'])
def build_kg():
    """
    构建并可视化知识图谱接口
    输入：JSON 数据，包含 base_path（数据集路径）或 text（文本内容）和 doc_id（文档 ID）
    输出：JSON 数据，包含可视化成功的消息
    """
    data = request.json
    text = data.get('text')  # 获取文本内容
    doc_id = data.get('doc_id')  # 获取文档 ID

    if not text or not doc_id:
        return jsonify({"error": "text and doc_id are required"}), 400

    try:
        # 构建知识图谱
        kg_ent, kg_rel = build(text, doc_id)

        # 可视化知识图谱
        result = to_neo4j(entities=kg_ent, relationships=kg_rel)

        # 返回成功消息
        return jsonify({
            # "doc_id": doc_id,
            "message": "Knowledge Graph built and visualized successfully",
            # "entities": kg_ent,
            # "relationships": kg_rel
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)