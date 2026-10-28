"""
配置模块：管理项目所有配置项，支持从环境变量和.env文件加载
"""
import os
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

class Config:
    """全局配置类"""
    
    # ========== 大模型API配置 ==========
    # 支持OpenAI兼容的API（如DeepSeek、智谱AI、通义千问等）
    LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")
    
    # ========== Embedding配置 ==========
    # 选择Embedding方式："api" 使用API调用，"local" 使用本地开源模型
    EMBEDDING_TYPE = os.getenv("EMBEDDING_TYPE", "api")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-ada-002")
    # 本地Embedding模型名称（仅当EMBEDDING_TYPE=local时生效）
    LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # ========== 向量数据库配置 ==========
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
    VECTOR_DB_COLLECTION = os.getenv("VECTOR_DB_COLLECTION", "rag_knowledge_base")
    
    # ========== 文档处理配置 ==========
    # 文本分割参数
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # ========== RAG配置 ==========
    # 检索返回的文档数量
    RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "3"))
    
    # ========== Agent配置 ==========
    # Agent最大迭代次数
    AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "5"))
    
    # ========== 应用配置 ==========
    APP_TITLE = "RAG智能问答Agent"
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# 全局配置实例
config = Config()
