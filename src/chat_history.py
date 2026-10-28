"""
对话历史管理模块：负责管理多轮对话的上下文
支持历史记录的添加、获取、清空，以及自动截断过长的历史
"""
from typing import List, Dict
from langchain.schema import HumanMessage, AIMessage, BaseMessage

from .config import config

class ChatHistoryManager:
    """对话历史管理器"""
    
    def __init__(self, max_history: int = 10):
        """
        初始化对话历史管理器
        
        Args:
            max_history: 最大保留的对话轮数（每轮包含用户和AI各一条）
        """
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []
    
    def add_user_message(self, message: str):
        """
        添加用户消息
        
        Args:
            message: 用户消息内容
        """
        self.history.append({"role": "user", "content": message})
        self._trim_history()
    
    def add_ai_message(self, message: str):
        """
        添加AI消息
        
        Args:
            message: AI消息内容
        """
        self.history.append({"role": "assistant", "content": message})
        self._trim_history()
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        获取完整对话历史
        
        Returns:
            对话历史列表
        """
        return self.history.copy()
    
    def get_langchain_messages(self) -> List[BaseMessage]:
        """
        获取LangChain格式的消息列表
        
        Returns:
            LangChain消息对象列表
        """
        messages = []
        for msg in self.history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        return messages
    
    def get_history_text(self) -> str:
        """
        获取文本格式的对话历史
        
        Returns:
            格式化的对话历史文本
        """
        history_text = ""
        for msg in self.history:
            if msg["role"] == "user":
                history_text += f"用户: {msg['content']}\n"
            elif msg["role"] == "assistant":
                history_text += f"助手: {msg['content']}\n"
        return history_text
    
    def clear(self):
        """清空对话历史"""
        self.history = []
    
    def _trim_history(self):
        """
        裁剪对话历史，保留最近的max_history轮
        每轮包含用户和AI各一条消息，所以总消息数是max_history * 2
        """
        max_messages = self.max_history * 2
        if len(self.history) > max_messages:
            # 保留最新的max_messages条
            self.history = self.history[-max_messages:]
    
    def get_history_count(self) -> int:
        """
        获取对话轮数
        
        Returns:
            对话轮数
        """
        return len(self.history) // 2
