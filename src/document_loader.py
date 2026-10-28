"""
文档加载模块：负责加载和处理不同格式的文档，进行文本分割
支持PDF、TXT格式
"""
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from .config import config

class DocumentLoader:
    """文档加载器类"""
    
    def __init__(self):
        """初始化文档加载器"""
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
            length_function=len
        )
    
    def load_file(self, file_path: str) -> List[Document]:
        """
        加载单个文档文件
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            分割后的文档列表
            
        Raises:
            ValueError: 不支持的文件格式
            FileNotFoundError: 文件不存在
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 获取文件扩展名
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # 根据文件格式选择加载器
        if file_ext == ".pdf":
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif file_ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}，仅支持PDF和TXT格式")
        
        # 对文档进行分割
        split_docs = self.text_splitter.split_documents(documents)
        
        # 为每个文档添加元数据
        for i, doc in enumerate(split_docs):
            doc.metadata["source"] = os.path.basename(file_path)
            doc.metadata["chunk_id"] = i
        
        return split_docs
    
    def load_files(self, file_paths: List[str]) -> List[Document]:
        """
        批量加载多个文档文件
        
        Args:
            file_paths: 文档文件路径列表
            
        Returns:
            所有分割后的文档列表
        """
        all_documents = []
        for file_path in file_paths:
            try:
                docs = self.load_file(file_path)
                all_documents.extend(docs)
            except Exception as e:
                print(f"加载文件 {file_path} 失败: {str(e)}")
                continue
        
        return all_documents
    
    def load_directory(self, dir_path: str) -> List[Document]:
        """
        加载目录下所有支持的文档
        
        Args:
            dir_path: 目录路径
            
        Returns:
            所有分割后的文档列表
        """
        if not os.path.isdir(dir_path):
            raise ValueError(f"目录不存在: {dir_path}")
        
        file_paths = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith((".pdf", ".txt")):
                    file_paths.append(os.path.join(root, file))
        
        return self.load_files(file_paths)
