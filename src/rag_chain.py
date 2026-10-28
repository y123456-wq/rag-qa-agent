"""
RAG链模块：实现检索增强生成的核心逻辑
将向量检索与大模型生成结合，实现基于知识库的智能问答
"""
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from .config import config
from .vector_store import VectorStore

class RAGChain:
    """RAG问答链类"""
    
    def __init__(self, vector_store: VectorStore):
        """
        初始化RAG链
        
        Args:
            vector_store: 向量数据库实例
        """
        self.vector_store = vector_store
        
        # 初始化大模型
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL_NAME,
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_API_BASE,
            temperature=0.7,
            streaming=True
        )
        
        # 构建RAG提示词模板
        self.prompt_template = self._build_prompt_template()
        
        # 构建检索QA链
        self.qa_chain = self._build_qa_chain()
    
    def _build_prompt_template(self) -> PromptTemplate:
        """
        构建RAG提示词模板
        
        Returns:
            提示词模板实例
        """
        template = """你是一个专业的智能问答助手，需要基于提供的上下文信息回答用户的问题。

请严格遵循以下规则：
1. 仅使用上下文信息中明确提到的内容回答问题
2. 如果上下文信息中没有相关内容，请明确说明"根据现有知识库无法回答该问题"
3. 回答要准确、简洁、有条理
4. 不要编造或猜测信息
5. 可以引用上下文的原文内容，但要注明来源

上下文信息：
{context}

用户问题：{question}

请给出你的回答："""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _build_qa_chain(self) -> RetrievalQA:
        """
        构建检索QA链
        
        Returns:
            RetrievalQA链实例
        """
        # 创建检索器
        retriever = self.vector_store.vector_store.as_retriever(
            search_kwargs={"k": config.RETRIEVE_TOP_K}
        )
        
        # 构建RetrievalQA链
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": self.prompt_template
            }
        )
        
        return qa_chain
    
    def query(self, question: str) -> dict:
        """
        执行RAG问答
        
        Args:
            question: 用户问题
            
        Returns:
            包含回答和来源文档的字典
        """
        # 执行查询
        result = self.qa_chain.invoke({"query": question})
        
        # 整理结果
        answer = result["result"]
        source_docs = result["source_documents"]
        
        # 提取来源信息
        sources = []
        for doc in source_docs:
            sources.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "未知来源"),
                "page": doc.metadata.get("page", None)
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "source_count": len(sources)
        }
    
    def get_retrieved_docs(self, query: str) -> List[Document]:
        """
        获取检索到的文档（用于调试或展示）
        
        Args:
            query: 查询文本
            
        Returns:
            检索到的文档列表
        """
        return self.vector_store.similarity_search(query)
