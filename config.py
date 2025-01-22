import os

llm_name = "qwen2:7b-ctx32k"
model_path = "/root/chatglm3-6b"

zhipu_model_name = "glm-4-flash"
zhipu_api_key = ''

# es_ip = os.environ.get('ES_IP', "")
es_ip = ""
es_doc_index = "idmms_split_file_info"
# es_vector_index = "medical_base"
es_vector_index = "education_base"

# ollama_url = os.environ.get('OLLAMA_URL', "")
ollama_url = ""
ollama_embeddings = "quentinz/bge-large-zh-v1.5"
local_embeddings = "/root/bge-large-zh-v1.5"

# neo4j
## 文档结构图谱
neo4j_ip = ""
neo4j_local_ip = "b"

## 医学专题图谱
# neo4j_ip = "bolt://192.168.88.2:7687"
neo4j_user = "neo4j"
neo4j_password = ""

chunk_size = 6000