import os
import sys
import time
from datetime import datetime
import streamlit as st
from pathlib import Path
import pdfplumber 
from streamlit_pdf_viewer import pdf_viewer


sys.path.insert(0, str(Path(__file__).parent / "src"))

# 尝试导入后端模块，如果报错给提示
try:
    from processor import FinancialDataProcessor
    from embedder import FinancialVectorDB
    from retrieval import DocumentRetriever
    from generator import ResponseGenerator
except ImportError as e:
    st.error(f"无法导入后端模块: {e}")
    st.info("请确保 src 文件夹下包含 processor.py, embedder.py, retrieval.py, generator.py")
    st.stop()

# ==================== 1. 页面配置 ====================
st.set_page_config(
    page_title="多模态文档理解助手",
    page_icon="FinTrace",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 2. CSS 样式 ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    .stApp { background-color: #f8fafc; font-family: 'Inter', sans-serif; color: #0f172a; }
    
    /* 侧边栏 */
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }

    /* 聊天气泡 */
    .stChatMessage { background: transparent !important; border: none !important; }
    div[data-testid="chatAvatarIcon-user"] { background-color: #6366f1 !important; }
    div[data-testid="chatAvatarIcon-assistant"] { background-color: #10b981 !important; }

    /* 来源按钮 (胶囊风) */
    .stButton > button {
        background-color: #f1f5f9; color: #475569; border: 1px solid #cbd5e1;
        border-radius: 99px; font-size: 12px; padding: 4px 12px; transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #3b82f6; color: white; border-color: #3b82f6;
    }

    /* 标题区域 */
    .app-header { margin-bottom: 20px; text-align: left; }
    .app-title {
        font-size: 26px; font-weight: 700;
        background: linear-gradient(90deg, #2563eb, #db2777);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .app-desc { color: #64748b; font-size: 14px; margin-top: 5px; }
    
    /* 隐藏杂项 */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==================== 3. 核心工具：高亮坐标计算器 ====================
def get_pdf_highlights(pdf_path, page_num, text_to_find):
    """
    使用 pdfplumber 在指定页面搜索文本，返回高亮坐标 (annotations)
    """
    annotations = []
    if not text_to_find or len(text_to_find) < 2: 
        return [] 

    try:
        with pdfplumber.open(pdf_path) as pdf:
            # pdfplumber 的页码从 0 开始
            if page_num - 1 < len(pdf.pages):
                page = pdf.pages[page_num - 1] 
                
                # 搜索文本
                words = page.search(text_to_find)
                
                # 构造 streamlit-pdf-viewer 需要的格式
                for word in words:
                    annotations.append({
                        "page": page_num,
                        "x": word["x0"],
                        "y": word["top"],
                        "width": word["x1"] - word["x0"],
                        "height": word["bottom"] - word["top"],
                        "color": "rgba(255, 255, 0, 0.4)", # 黄色半透明
                        "cursor": "pointer"
                    })
    except Exception as e:
        print(f"高亮计算失败: {e}")
    
    return annotations

# ====================  引擎加载与业务逻辑 ====================
@st.cache_resource(show_spinner=True)
def load_engine():
    """缓存检索器和生成器，避免重复初始化"""
    try:
        with st.spinner("正在初始化 AI 引擎（首次加载可能需要一些时间）..."):
            retriever = DocumentRetriever(db_path="data/vector_db")
            generator = ResponseGenerator()
        return retriever, generator
    except Exception as e:
        st.error(f"引擎初始化失败: {e}")
        st.info("提示：请检查向量数据库是否存在，或先上传文档进行处理")
        raise

@st.cache_resource(show_spinner=False)
def get_processor():
    """缓存处理器实例"""
    return FinancialDataProcessor(input_dir="my_rag_data", output_dir="data/processed")


def get_basic_kpis():
    """
    简单 KPI 统计，用于页面头部展示：
    - 已上传文档数（my_rag_data/uploaded 下的文件）
    - 已处理文档数（data/processed 下的 JSON 文件）
    """
    upload_dir = Path("my_rag_data/uploaded")
    processed_dir = Path("data/processed")

    try:
        uploaded_files = [
            p for p in upload_dir.glob("*")
            if p.is_file() and not p.name.startswith(".")
        ]
        uploaded_count = len(uploaded_files)
    except Exception:
        uploaded_count = 0

    try:
        processed_files = list(processed_dir.glob("*.json"))
        processed_count = len(processed_files)
    except Exception:
        processed_count = 0

    return uploaded_count, processed_count


def export_chat_markdown(messages):
    """将当前问答记录导出为可复用的 Markdown 研究笔记。"""
    lines = ["# FinLens RAG 对话记录", "", f"> 导出时间：{datetime.now():%Y-%m-%d %H:%M}", ""]
    for message in messages:
        title = "用户问题" if message["role"] == "user" else "AI 回答"
        lines.extend([f"## {title}", "", message["content"], ""])
        if message.get("sources"):
            lines.extend(["**参考来源**", ""])
            for source in message["sources"]:
                lines.append(f"- {source.get('source', '未知文件')}，第 {source.get('page', 1)} 页")
            lines.append("")
    return "\n".join(lines)

def sync_data(uploaded_files):
    """优化的数据同步函数：支持增量处理和进度显示"""
    upload_dir = Path("my_rag_data/uploaded")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # 计算总文件大小，用于进度估算
    total_size = sum(f.size for f in uploaded_files)
    
    with st.status("知识库同步中...", expanded=True) as status:
        # 步骤1: 保存上传的文件
        status.update(label="正在保存文件...", state="running")
        saved_files = []
        progress_bar = st.progress(0)
        
        for idx, f in enumerate(uploaded_files):
            file_path = upload_dir / f.name
            with open(file_path, "wb") as save_f: 
                save_f.write(f.getbuffer())
            saved_files.append(file_path)
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        # 步骤2: 处理文件（只处理新文件）
        status.update(label="正在解析文档...", state="running")
        processor = get_processor()
        processor.process_all(only_new_files=True)  # 只处理新文件
        
        # 步骤3: 向量化（增量更新）
        status.update(label="正在生成向量索引（这可能需要几分钟，请耐心等待）...", state="running")
        # 使用缓存的数据库实例（但需要重新初始化以加载最新数据）
        db = FinancialVectorDB(input_dir="data/processed", persist_dir="data/vector_db")
        
        # 显示详细进度提示
        st.info("向量化过程说明：\n"
               "- 正在将文档转换为向量表示\n"
               "- 每个文档片段需要生成嵌入向量\n"
               "- 处理速度取决于文档数量和硬件性能\n"
               "- 请保持页面打开，不要关闭浏览器")
        
        db.build_db(clear_existing=False, incremental=True)  # 增量更新
        
        status.update(label="同步完成", state="complete", expanded=False)
        progress_bar.empty()
        st.toast(f"知识库已更新！已处理 {len(saved_files)} 个文件")

# Session 初始化
if "messages" not in st.session_state: st.session_state.messages = []
if "view_pdf" not in st.session_state: st.session_state.view_pdf = None
if "view_page" not in st.session_state: st.session_state.view_page = 1
if "highlight_text" not in st.session_state: st.session_state.highlight_text = ""
if "view_text_file" not in st.session_state: st.session_state.view_text_file = None
if "view_text_content" not in st.session_state: st.session_state.view_text_content = ""

# ==================== 5. 工具函数 ====================

def _resolve_source_path(source_name: str):
    """
    根据 metadata 中的 source 字段，在本地寻找真实文件路径。
    兼容：
    - 直接存原始文件名（如 xxx.pdf）
    - 存处理后的 .json 文件名，需要还原为原始扩展名
    """
    if not source_name:
        return None
    
    original_name = source_name
    possible_paths = [
        f"my_rag_data/uploaded/{original_name}",
        f"my_rag_data/{original_name}",
        source_name,  # 可能已经是绝对路径
    ]
    
    # 如果是 .json，尝试还原为原始格式
    if original_name.endswith(".json"):
        base_name = original_name.replace(".json", "")
        for ext in [".pdf", ".docx", ".txt", ".html", ".htm"]:
            possible_paths.extend(
                [
                    f"my_rag_data/uploaded/{base_name}{ext}",
                    f"my_rag_data/{base_name}{ext}",
                ]
            )
    
    return next((p for p in possible_paths if os.path.exists(p)), None)


def _open_source_in_viewer(source: dict):
    """
    根据单个 source 记录，自动在左侧原文视图中打开并高亮。
    用于：
    - 用户点击【来源】按钮
    - 首轮自动溯源高亮
    """
    real_path = _resolve_source_path(source.get("source"))
    if not real_path:
        return False

    page = int(source.get("page", 1))
    text = source.get("text", "")
    file_ext = Path(real_path).suffix.lower()

    if file_ext == ".pdf":
        st.session_state.view_pdf = real_path
        st.session_state.view_page = page
        st.session_state.highlight_text = text
        # 清空文本视图
        st.session_state.view_text_file = None
        st.session_state.view_text_content = ""
    else:
        st.session_state.view_pdf = None
        st.session_state.view_text_file = real_path
        st.session_state.view_text_content = text
        st.session_state.view_page = page
        st.session_state.highlight_text = text

    return True


# ====================  主界面布局 ====================

# --- 侧边栏 ---
with st.sidebar:
    st.title("FinRAG Pro")
    st.caption("文档理解分析系统")
    st.divider()
    
    files = st.file_uploader(
        "上传文档 (支持 PDF/Word/TXT/HTML)", 
        type=["pdf", "docx", "doc", "txt", "html", "htm"], 
        accept_multiple_files=True, 
        label_visibility="collapsed",
        help="支持格式: PDF (.pdf), Word (.docx), 文本 (.txt), HTML (.html, .htm)"
    )
    if st.button("开始处理", use_container_width=True, type="primary"):
        if files: 
            # 检查文件类型并给出提示
            unsupported = [f.name for f in files if not f.name.lower().endswith(('.pdf', '.docx', '.txt', '.html', '.htm'))]
            if unsupported:
                st.warning(f"以下文件格式暂不支持: {', '.join(unsupported)}")
            sync_data(files)
        else: 
            st.warning("请先选择文件")
        
    st.divider()
    uploaded_documents = sorted(
        p.name for p in Path("my_rag_data/uploaded").glob("*") if p.is_file()
    )
    selected_document = st.selectbox(
        "检索范围",
        options=["全部文档", *uploaded_documents],
        help="选择单个文档时，问答只从该文档中检索，便于聚焦核验。",
    )
    if st.session_state.messages:
        st.download_button(
            "导出当前对话",
            data=export_chat_markdown(st.session_state.messages),
            file_name=f"finlens-chat-{datetime.now():%Y%m%d-%H%M}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    st.divider()
    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.view_pdf = None
        st.session_state.view_text_file = None
        st.session_state.view_text_content = ""
        st.session_state.highlight_text = ""
        st.rerun()

# --- 顶部 ---
st.markdown("""
<div class="app-header">
    <div class="app-title">文档理解平台</div>
    <div class="app-desc">
        提供高置信度回答与一键原文溯源。
    </div>
</div>
""", unsafe_allow_html=True)


uploaded_count, processed_count = get_basic_kpis()
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
with kpi_col1:
    st.metric("已上传文档数", uploaded_count)
with kpi_col2:
    st.metric("已纳入知识库片段数", processed_count)
with kpi_col3:
    st.metric("当前会话轮次", len([m for m in st.session_state.messages if m["role"] == "user"]))


col_pdf, col_chat = st.columns([1.4, 1], gap="large")

# === 目前 (支持 PDF/Word/TXT/HTML) ===
with col_pdf:
    with st.container():
        st.markdown("**原文查证**")
        
        # 优先显示 PDF
        if st.session_state.view_pdf and os.path.exists(st.session_state.view_pdf):
            pdf_path = st.session_state.view_pdf
            pdf_name = Path(pdf_path).name
            page_num = int(st.session_state.view_page) # 确保 int
            target_text = st.session_state.highlight_text
            
            st.caption(f"当前文件: {pdf_name} | 第 {page_num} 页")
            
            # 计算高亮坐标
            annotations = []
            if target_text:
                # 截取前 50 个字符进行搜索
                search_snippet = target_text[:50] 
                annotations = get_pdf_highlights(pdf_path, page_num, search_snippet)
                if not annotations:
                    st.toast("文本匹配度较低，显示当前页，暂无高亮")

        
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            

            viewer_key = f"pdf_viewer_{pdf_name}_{page_num}_{int(time.time())}"
            
            pdf_viewer(
                input=pdf_bytes,
                width=800,
                height=800,
                scroll_to_page=page_num,  # 这里的页码必须对应 key 的变化
                annotations=annotations,  # 传入高亮
                render_text=True,
                key=viewer_key      
            )
              

        
        # 显示非 PDF 文件（Word/TXT/HTML）的文本预览
        elif st.session_state.view_text_file and os.path.exists(st.session_state.view_text_file):
            text_file_path = st.session_state.view_text_file
            file_name = Path(text_file_path).name
            page_num = int(st.session_state.view_page)
            file_ext = Path(text_file_path).suffix.lower()
            
            # 根据文件类型显示不同的图标和标题
            file_type_names = {
                '.docx': 'Word 文档',
                '.doc': 'Word 文档 (旧格式)',
                '.txt': '文本文件',
                '.html': 'HTML 文件',
                '.htm': 'HTML 文件'
            }
            file_type_name = file_type_names.get(file_ext, '文档')
            
            st.caption(f"当前文件: {file_name} ({file_type_name}) | 第 {page_num} 页/段")
            
            # 显示文本内容（带高亮）
            text_content = st.session_state.view_text_content
            if text_content:
                # 尝试高亮目标文本
                highlight_text = st.session_state.highlight_text[:50] if st.session_state.highlight_text else ""
                if highlight_text and highlight_text in text_content:
                    # 使用 HTML 标记高亮
                    highlighted_content = text_content.replace(
                        highlight_text, 
                        f'<mark style="background-color: yellow;">{highlight_text}</mark>'
                    )
                    st.markdown(f'<div style="background: white; padding: 20px; border-radius: 8px; border: 1px solid #e2e8f0; max-height: 700px; overflow-y: auto; font-family: monospace; white-space: pre-wrap;">{highlighted_content}</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="background: white; padding: 20px; border-radius: 8px; border: 1px solid #e2e8f0; max-height: 700px; overflow-y: auto; font-family: monospace; white-space: pre-wrap;">{text_content}</div>', 
                              unsafe_allow_html=True)
            else:
                # 如果 session 中没有内容，尝试读取文件
                try:
                    if file_ext == '.docx':
                        st.info("Word 文档预览：显示检索到的相关段落。完整文档请下载查看。")
                    else:
                        with open(text_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            st.text_area("文档内容", content, height=700, disabled=True)
                except Exception as e:
                    st.error(f"读取文件失败: {e}")
        
        else:
            # 占位图
            st.markdown("""
            <div style="background: white; border-radius: 12px; border: 2px dashed #cbd5e1; height: 600px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                <h3 style="color: #94a3b8;">原文视窗</h3>
                <p style="color: #cbd5e1; font-size: 14px;">提问后点击【来源按钮】，此处将自动跳转并高亮</p>
                <p style="color: #cbd5e1; font-size: 12px; margin-top: 10px;">支持 PDF、Word、TXT、HTML 等格式</p>
            </div>
            """, unsafe_allow_html=True)

# === 右侧：对话 ===
with col_chat:
    with st.container():
        st.markdown("**AI 分析师**")
        chat_container = st.container(height=700)
        
        try:
            retriever, generator = load_engine()
        except Exception as e:
            st.error(f"AI 引擎初始化失败: {e}")
            st.info("可能的原因：\n"
                   "1. 向量数据库不存在，请先上传文档进行处理\n"
                   "2. 模型加载失败，请检查网络连接或模型文件\n"
                   "3. GPU 资源不足，尝试使用 CPU 模式")
            st.stop()
            
        with chat_container:
            if not st.session_state.messages:
                st.info("欢迎！请上传文件并开始提问。")

            for idx, msg in enumerate(st.session_state.messages):
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    
                    # 来源按钮逻辑
                    if "sources" in msg and msg["sources"]:
                        st.markdown("")
                        # 动态计算列数，最多3列一行
                        n_sources = len(msg["sources"])
                        cols = st.columns(min(n_sources, 3))
                        
                        for i, src in enumerate(msg["sources"]):
                            # 处理文件名：支持多种文件类型
                            source_name = src['source']
                            real_path = _resolve_source_path(source_name)
                            source_text = src.get('text', '') # 获取用于高亮的文本
                            
                            # 判断文件类型
                            file_ext = Path(real_path).suffix.lower() if real_path else None
                            is_pdf = file_ext == '.pdf' if file_ext else False

                            # 防止列溢出
                            col_idx = i % 3
                            with cols[col_idx]:
                                # 根据文件类型显示不同的图标
                                btn_label = f"P{src.get('page', 1)} 来源"

                                if st.button(btn_label, key=f"btn_{idx}_{i}"):
                                    if real_path and _open_source_in_viewer(src):
                                        st.rerun()  
                                    else:
                                        st.error(f"未找到文件: {source_name}")

        if prompt := st.chat_input("输入问题..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"): st.write(prompt)
                
                with st.chat_message("assistant"):
                    # 使用占位符实现流式输出效果
                    answer_placeholder = st.empty()
                    
                    with st.status("检索与分析中...", expanded=False) as status:
                        # 1. 检索
                        status.update(label=" 正在检索相关文档...", state="running")
                        source_filter = None if selected_document == "全部文档" else selected_document
                        docs = retriever.retrieve_and_rerank(
                            prompt, top_k=3, source_filter=source_filter
                        )
                        
                        # 2. 生成（显示加载动画）
                        status.update(label=" AI 正在分析并生成回答...", state="running")
                        answer = generator.generate(prompt, docs)
                    
                    # 3. 格式化显示回答（支持 Markdown）
                    answer_placeholder.markdown(answer)
                    
                    # 4. 构造来源数据 (包含 text 用于高亮)
                    sources = []
                    for d in docs:
                        meta = d.get('metadata', {})
                        sources.append({
                            "source": meta.get('source', 'unknown.pdf'),
                            "page": meta.get('page', 1),
                            "text": d.get('text', '') # 确保这里有文本
                        })

                    # 5. 回答完成后自动在左侧打开首个来源并高亮
                    if sources:
                        _open_source_in_viewer(sources[0])
                    
                    # 6. 显示来源提示
                    if sources:
                        st.caption(f"参考了 {len(sources)} 个文档片段，点击下方按钮查看原文")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    st.rerun()
