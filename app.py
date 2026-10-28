"""
RAG智能问答Agent - Gradio图形界面
提供文档上传、知识库管理、智能对话等功能
"""
import os
import gradio as gr
from src import (
    DocumentLoader,
    VectorStore,
    ChatHistoryManager,
    RAGAgent,
    config
)

# 全局实例
document_loader = DocumentLoader()
vector_store = VectorStore()
chat_history = ChatHistoryManager(max_history=10)
rag_agent = RAGAgent(vector_store, chat_history)

# 临时文件存储目录
TEMP_DIR = "./data/temp"
os.makedirs(TEMP_DIR, exist_ok=True)


def upload_files(files):
    """
    处理文件上传，构建知识库
    
    Args:
        files: 上传的文件列表
        
    Returns:
        处理结果信息，知识库状态
    """
    if not files:
        return "请先选择要上传的文件", get_kb_status()
    
    try:
        # 保存上传的文件到临时目录
        file_paths = []
        for file in files:
            # 获取文件名
            filename = os.path.basename(file.orig_name)
            save_path = os.path.join(TEMP_DIR, filename)
            
            # 保存文件
            with open(save_path, "wb") as f:
                f.write(file.read())
            
            file_paths.append(save_path)
        
        # 加载文档
        documents = document_loader.load_files(file_paths)
        
        if not documents:
            return "没有成功加载任何文档，请检查文件格式", get_kb_status()
        
        # 添加到向量数据库
        added_count = vector_store.add_documents(documents)
        
        result = f"✅ 成功添加 {added_count} 个文档片段到知识库\n"
        result += f"📄 处理了 {len(file_paths)} 个文件\n"
        result += f"📚 当前知识库总片段数：{vector_store.get_document_count()}"
        
        return result, get_kb_status()
    
    except Exception as e:
        return f"❌ 处理文件失败：{str(e)}", get_kb_status()


def get_kb_status():
    """获取知识库状态"""
    try:
        doc_count = vector_store.get_document_count()
        sources = vector_store.get_all_sources()
        
        status = f"📊 知识库状态\n"
        status += f"━━━━━━━━━━━━━━━━\n"
        status += f"文档片段总数：{doc_count} 个\n"
        status += f"文档来源数：{len(sources)} 个文件\n\n"
        
        if sources:
            status += "📁 已加载的文档：\n"
            for source in sources:
                status += f"  • {source}\n"
        
        return status
    except Exception as e:
        return f"获取知识库状态失败：{str(e)}"


def clear_knowledge_base():
    """清空知识库"""
    try:
        vector_store.clear()
        chat_history.clear()
        return "✅ 知识库已清空，对话历史已重置", get_kb_status(), []
    except Exception as e:
        return f"❌ 清空失败：{str(e)}", get_kb_status(), []


def chat_with_agent(message, history, show_steps):
    """
    与Agent对话
    
    Args:
        message: 用户消息
        history: 对话历史
        show_steps: 是否显示思考步骤
        
    Returns:
        更新后的对话历史
    """
    if not message.strip():
        return history
    
    # 检查知识库是否为空
    doc_count = vector_store.get_document_count()
    if doc_count == 0:
        response = "⚠️ 知识库当前为空，请先上传文档构建知识库后再提问哦～"
        history.append((message, response))
        return history
    
    try:
        # 调用Agent对话
        result = rag_agent.chat(message)
        
        response = result["answer"]
        
        # 如果显示思考步骤，添加到回复中
        if show_steps and result["steps"]:
            steps_text = "\n\n🤔 思考过程：\n"
            steps_text += "━━━━━━━━━━━━━━━━\n"
            for i, step in enumerate(result["steps"], 1):
                steps_text += f"\n步骤 {i}：\n"
                steps_text += f"🔧 调用工具：{step['tool']}\n"
                steps_text += f"📝 工具输入：{step['tool_input']}\n"
                steps_text += f"💭 思考：{step['log'][:200]}...\n"
                steps_text += f"👁️ 观察结果：{step['observation'][:300]}...\n"
            
            response += steps_text
        
        history.append((message, response))
        return history
    
    except Exception as e:
        error_response = f"❌ 对话出错：{str(e)}\n\n请检查API配置是否正确。"
        history.append((message, error_response))
        return history


def clear_chat():
    """清空对话历史"""
    chat_history.clear()
    return []


# 构建Gradio界面
def build_ui():
    """构建Gradio用户界面"""
    with gr.Blocks(title=f"{config.APP_TITLE} v{config.APP_VERSION}", theme=gr.themes.Soft()) as demo:
        # 标题区域
        gr.Markdown(f"""
        # 🤖 {config.APP_TITLE}
        ### 基于RAG技术的智能问答Agent | 支持文档上传、知识库构建、多轮对话
        
        <div style="color: #666; font-size: 0.9em;">
        技术栈：Python + LangChain + Chroma向量数据库 + Gradio | 支持ReAct Agent模式
        </div>
        """)
        
        with gr.Row():
            # 左侧：知识库管理区域
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("## 📚 知识库管理")
                
                # 文件上传
                file_upload = gr.File(
                    label="上传文档（支持PDF、TXT格式）",
                    file_types=[".pdf", ".txt"],
                    file_count="multiple"
                )
                
                upload_btn = gr.Button("📤 上传并构建知识库", variant="primary")
                
                # 知识库状态
                kb_status = gr.Textbox(
                    label="知识库状态",
                    value=get_kb_status(),
                    lines=12,
                    interactive=False
                )
                
                # 操作按钮
                with gr.Row():
                    clear_kb_btn = gr.Button("🗑️ 清空知识库", variant="stop")
                    refresh_btn = gr.Button("🔄 刷新状态")
                
                # 设置选项
                gr.Markdown("---")
                gr.Markdown("### ⚙️ 对话设置")
                show_steps = gr.Checkbox(
                    label="显示Agent思考过程",
                    value=False,
                    info="开启后可查看Agent的工具调用和推理步骤"
                )
                
                # 上传结果
                upload_result = gr.Textbox(
                    label="处理结果",
                    lines=3,
                    interactive=False
                )
            
            # 右侧：对话区域
            with gr.Column(scale=2, min_width=500):
                gr.Markdown("## 💬 智能对话")
                
                # 聊天界面
                chatbot = gr.Chatbot(
                    label="与Agent对话",
                    height=500,
                    bubble_full_width=False,
                    avatar_images=(None, "🤖")
                )
                
                # 输入框
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="输入你的问题，按回车发送...",
                        scale=4,
                        show_label=False,
                        container=False
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)
                
                # 清空对话按钮
                clear_chat_btn = gr.Button("🗑️ 清空对话历史", size="sm")
                
                # 使用提示
                gr.Markdown("""
                <div style="color: #888; font-size: 0.85em; margin-top: 10px;">
                💡 使用提示：
                1. 先在左侧上传PDF或TXT文档构建知识库
                2. 然后就可以向Agent提问关于文档的问题
                3. 开启"显示思考过程"可查看Agent的推理过程
                4. 支持多轮对话，Agent会记住上下文
                </div>
                """)
        
        # 底部信息
        gr.Markdown("---")
        gr.Markdown(f"""
        <div style="text-align: center; color: #999; font-size: 0.8em;">
        RAG智能问答Agent v{config.APP_VERSION} | 程序设计实践课程项目
        </div>
        """)
        
        # 事件绑定
        upload_btn.click(
            fn=upload_files,
            inputs=[file_upload],
            outputs=[upload_result, kb_status]
        )
        
        refresh_btn.click(
            fn=get_kb_status,
            outputs=[kb_status]
        )
        
        clear_kb_btn.click(
            fn=clear_knowledge_base,
            outputs=[upload_result, kb_status, chatbot]
        )
        
        send_btn.click(
            fn=chat_with_agent,
            inputs=[msg_input, chatbot, show_steps],
            outputs=[chatbot]
        )
        
        msg_input.submit(
            fn=chat_with_agent,
            inputs=[msg_input, chatbot, show_steps],
            outputs=[chatbot]
        ).then(
            fn=lambda: "",
            outputs=[msg_input]
        )
        
        clear_chat_btn.click(
            fn=clear_chat,
            outputs=[chatbot]
        )
    
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=config.DEBUG
    )
