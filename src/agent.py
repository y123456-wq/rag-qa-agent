"""
Agent模块：实现智能Agent核心逻辑
采用ReAct（推理+行动）模式，支持自主调用工具完成复杂任务
包含知识库检索、信息总结等工具，体现Agent式对话设计
"""
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish

from .config import config
from .vector_store import VectorStore
from .rag_chain import RAGChain
from .chat_history import ChatHistoryManager

class RAGAgent:
    """RAG智能Agent类"""
    
    def __init__(self, vector_store: VectorStore, chat_history: ChatHistoryManager):
        """
        初始化RAG Agent
        
        Args:
            vector_store: 向量数据库实例
            chat_history: 对话历史管理器
        """
        self.vector_store = vector_store
        self.chat_history = chat_history
        self.rag_chain = RAGChain(vector_store)
        
        # 初始化大模型
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL_NAME,
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_API_BASE,
            temperature=0.7,
            streaming=True
        )
        
        # 创建工具列表
        self.tools = self._create_tools()
        
        # 构建Agent
        self.agent = self._build_agent()
        
        # 创建Agent执行器
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=config.DEBUG,
            max_iterations=config.AGENT_MAX_ITERATIONS,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    
    def _create_tools(self) -> List[Tool]:
        """
        创建Agent可用的工具列表
        
        Returns:
            工具列表
        """
        tools = [
            Tool(
                name="知识库检索",
                func=self._knowledge_base_search,
                description="""当需要回答基于知识库的专业问题、查找文档中的具体信息时使用此工具。
                输入是用户的问题，输出是从知识库中检索到的相关信息和答案。
                当问题涉及文档内容、专业知识、具体事实时，优先使用此工具。"""
            ),
            Tool(
                name="知识库统计",
                func=self._knowledge_base_stats,
                description="""当需要了解知识库的基本信息时使用此工具。
                不需要输入参数，返回知识库中的文档数量、文档来源等统计信息。
                当用户询问"知识库中有什么"、"有多少文档"等问题时使用。"""
            )
        ]
        
        return tools
    
    def _knowledge_base_search(self, query: str) -> str:
        """
        知识库检索工具函数
        
        Args:
            query: 查询问题
            
        Returns:
            检索结果和答案
        """
        try:
            result = self.rag_chain.query(query)
            
            # 格式化结果
            output = f"回答：{result['answer']}\n\n"
            output += f"参考来源（共{result['source_count']}个）：\n"
            for i, source in enumerate(result['sources'], 1):
                output += f"{i}. 来源：{source['source']}"
                if source['page'] is not None:
                    output += f"，第{source['page']}页"
                output += "\n"
                output += f"   内容摘要：{source['content'][:200]}...\n"
            
            return output
        except Exception as e:
            return f"检索失败：{str(e)}"
    
    def _knowledge_base_stats(self, _: str = "") -> str:
        """
        知识库统计工具函数
        
        Args:
            _: 忽略的输入参数
            
        Returns:
            知识库统计信息
        """
        try:
            doc_count = self.vector_store.get_document_count()
            sources = self.vector_store.get_all_sources()
            
            output = f"知识库统计信息：\n"
            output += f"- 文档片段总数：{doc_count}个\n"
            output += f"- 文档来源（共{len(sources)}个文件）：\n"
            for source in sources:
                output += f"  - {source}\n"
            
            return output
        except Exception as e:
            return f"获取统计信息失败：{str(e)}"
    
    def _build_agent(self):
        """
        构建ReAct Agent
        
        Returns:
            Agent实例
        """
        # ReAct提示词模板
        template = """你是一个专业的智能助手，拥有自主思考和工具调用能力。你可以通过调用工具来获取信息，然后回答用户的问题。

你可以使用以下工具：
{tools}

工具名称列表：{tool_names}

请严格按照以下格式进行思考和行动：

问题：用户的输入问题
思考：你需要思考下一步该做什么，是否需要调用工具
行动：工具名称，必须是 [{tool_names}] 中的一个
行动输入：工具的输入参数
观察：工具返回的结果
...（这个思考/行动/观察的过程可以重复多次）
思考：现在我已经有足够的信息来回答用户的问题了
最终回答：给出最终的回答

注意事项：
1. 当问题涉及知识库中的内容时，优先使用"知识库检索"工具
2. 当用户询问知识库的基本信息时，使用"知识库统计"工具
3. 如果工具返回的信息不足以回答问题，可以尝试调整查询词再次调用工具
4. 如果多次调用工具仍无法获得答案，请如实告知用户
5. 回答要准确、简洁、有条理
6. 结合对话历史理解用户的问题

对话历史：
{chat_history}

当前问题：{input}

{agent_scratchpad}"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "chat_history", "agent_scratchpad"]
        )
        
        # 创建ReAct Agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return agent
    
    def chat(self, user_input: str) -> Dict[str, Any]:
        """
        与Agent进行对话
        
        Args:
            user_input: 用户输入
            
        Returns:
            包含回答和中间步骤的字典
        """
        try:
            # 获取对话历史
            chat_history_text = self.chat_history.get_history_text()
            
            # 执行Agent
            result = self.agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history_text
            })
            
            # 提取结果
            answer = result["output"]
            intermediate_steps = result.get("intermediate_steps", [])
            
            # 整理中间步骤
            steps = []
            for step in intermediate_steps:
                action: AgentAction = step[0]
                observation = step[1]
                steps.append({
                    "tool": action.tool,
                    "tool_input": action.tool_input,
                    "log": action.log,
                    "observation": str(observation)[:500] + "..." if len(str(observation)) > 500 else str(observation)
                })
            
            # 更新对话历史
            self.chat_history.add_user_message(user_input)
            self.chat_history.add_ai_message(answer)
            
            return {
                "answer": answer,
                "steps": steps,
                "step_count": len(steps),
                "success": True
            }
            
        except Exception as e:
            error_msg = f"对话出错：{str(e)}"
            return {
                "answer": error_msg,
                "steps": [],
                "step_count": 0,
                "success": False
            }
    
    def clear_history(self):
        """清空对话历史"""
        self.chat_history.clear()
