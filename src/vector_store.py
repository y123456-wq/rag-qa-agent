"""
向量数据库模块：负责向量存储和检索
使用Chroma作为轻量级向量数据库，支持API和本地两种Embedding方式
"""
import os
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from .config import config

class VectorStore:
    """向量数据库类"""
    
    def __init__(self):
        """初始化向量数据库"""
        # 初始化Embedding模型
        self.embeddings = self._init_embeddings()
        
        # 初始化向量数据库
        self.vector_store = self._init_vector_store()
    
    def _init_embeddings(self):
        """
        初始化Embedding模型
        
        Returns:
            Embedding模型实例
        """
        if config.EMBEDDING_TYPE == "api":
            # 使用API方式的Embedding
            return OpenAIEmbeddings(
                model=config.EMBEDDING_MODEL_NAME,
                api_key=config.LLM_API_KEY,
                base_url=config.LLM_API_BASE
            )
        elif config.EMBEDDING_TYPE == "local":
            # 使用本地开源Embedding模型
            return HuggingFaceEmbeddings(
                model_name=config.LOCAL_EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        else:
            raise ValueError(f"不支持的Embedding类型: {config.EMBEDDING_TYPE}")
    
    def _init_vector_store(self) -> Chroma:
        """
        初始化向量数据库
        
        Returns:
            Chroma向量数据库实例
        """
        # 确保目录存在
        os.makedirs(config.VECTOR_DB_PATH, exist_ok=True)
        
        # 创建或加载Chroma向量数据库
        vector_store = Chroma(
            collection_name=config.VECTOR_DB_COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=config.VECTOR_DB_PATH
        )
        
        return vector_store
    
    def add_documents(self, documents: List[Document]) -> int:
        """
        向向量数据库添加文档
        
        Args:
            documents: 文档列表
            
        Returns:
            添加的文档数量
        """
        if not documents:
            return 0
        
        # 添加文档到向量数据库
        self.vector_store.add_documents(documents)
        
        # 持久化保存
        self.vector_store.persist()
        
        return len(documents)
    
    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        相似度检索
        
        Args:
            query: 查询文本
            k: 返回的文档数量，默认使用配置中的值
            
        Returns:
            检索到的文档列表
        """
        if k is None:
            k = config.RETRIEVE_TOP_K
        
        # 执行相似度检索
        results = self.vector_store.similarity_search(query, k=k)
        
        return results
    
    def similarity_search_with_score(self, query: str, k: Optional[int] = None) -> List[tuple]:
        """
        带分数的相似度检索
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            (文档, 分数) 元组列表
        """
        if k is None:
            k = config.RETRIEVE_TOP_K
        
        # 执行带分数的相似度检索
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        return results
    
    def get_document_count(self) -> int:
        """
        获取向量数据库中的文档数量
        
        Returns:
            文档数量
        """
        return self.vector_store._collection.count()
    
    def clear(self):
        """清空向量数据库"""
        # 删除集合
        self.vector_store.delete_collection()
        
        # 重新初始化
        self.vector_store = self._init_vector_store()
    
    def get_all_sources(self) -> List[str]:
        """
        获取所有文档来源
        
        Returns:
            文档来源文件名列表
        """
        # 获取所有文档的元数据
        results = self.vector_store.get(include=["metadatas"])
        
        sources = set()
        for metadata in results["metadatas"]:
            if "source" in metadata:
                sources.add(metadata["source"])
        
        return sorted(list(sources))
