# FinLens RAG

> 面向金融文档的可溯源智能问答系统

FinLens RAG 是一个基于检索增强生成（RAG）的智能文档分析应用，面向年报、研报、公告、制度文件等长文档场景。系统支持多格式文档解析、增量知识库构建、语义检索、重排序问答和原文溯源，帮助用户更快地定位信息、核验结论并沉淀研究记录。

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 功能特性

- **多格式文档解析**：支持 PDF、DOCX、TXT、HTML 文档上传、解析与清洗。
- **增量知识库构建**：仅处理新增或修改的文档，减少重复解析与向量化开销。
- **两阶段语义检索**：使用 BGE-M3 进行向量召回，并通过 BGE Reranker 对候选内容精排。
- **基于证据的问答**：大模型结合检索片段生成回答；信息不足时明确提示，减少无依据回答。
- **引用与原文溯源**：回答附带文件来源和页码，可跳转到 PDF 原页并高亮相关文本。
- **单文档检索**：可将问题限定在某一篇文档内，提高长文档核验效率。
- **对话记录导出**：支持将当前问答及引用来源导出为 Markdown 文件。
- **GPU / CPU 自适应**：优先使用可用 GPU；资源不足时自动回退至 CPU。

## 系统流程

```text
上传文档
   ↓
解析、清洗与分段
   ↓
向量化并写入 ChromaDB
   ↓
BGE-M3 语义召回
   ↓
BGE Reranker 重排序
   ↓
大模型基于检索证据生成回答
   ↓
展示来源、跳转原文并高亮核验
```

## 技术栈

| 模块 | 技术方案 |
| --- | --- |
| 应用界面 | Streamlit |
| 文档解析 | pdfplumber、python-docx、BeautifulSoup |
| RAG 框架 | LangChain |
| 向量数据库 | ChromaDB |
| 嵌入模型 | BAAI/bge-m3 |
| 重排序模型 | BAAI/bge-reranker-v2-m3 |
| 大语言模型 | OpenAI-compatible API（如 Qwen、DeepSeek） |
| 评估 | RAGAS |
| 部署 | Docker、Docker Compose |

## 项目结构

```text
finlens-rag/
├── app.py                  # Streamlit 应用入口
├── src/
│   ├── processor.py         # 文档解析、清洗与增量处理
│   ├── embedder.py          # 向量化与 ChromaDB 索引构建
│   ├── retrieval.py         # 向量检索、筛选与重排序
│   ├── generator.py         # 基于证据的回答生成
│   ├── parser.py            # 解析辅助工具
│   └── schema.py            # 数据模型定义
├── eval/
│   └── evaluate_rag.py      # RAGAS 评估脚本
├── data/
│   ├── processed/           # 处理后的文档数据（不提交）
│   └── vector_db/           # 本地向量数据库（不提交）
├── my_rag_data/
│   └── uploaded/            # 用户上传文件（不提交）
├── .env.example             # 环境变量模板
├── requirements.txt         # Python 依赖
├── Dockerfile
└── docker-compose.yml
```

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/<your-username>/finlens-rag.git
cd finlens-rag
```

### 2. 创建虚拟环境并安装依赖

```bash
python -m venv .venv
```

Windows：

```bash
.venv\\Scripts\\activate
pip install -r requirements.txt
```

macOS / Linux：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 配置模型服务

复制环境变量模板：

```bash
copy .env.example .env
```

macOS / Linux：

```bash
cp .env.example .env
```

编辑 `.env`，填入可用的 OpenAI-compatible API 配置：

```env
LLM_API_KEY=your_api_key_here
LLM_MODEL=qwen-max
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

> 请不要将 `.env`、上传文件、处理后的数据或向量数据库提交到 GitHub。

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器访问：`http://localhost:8501`

## 使用说明

1. 在左侧边栏上传 PDF、DOCX、TXT 或 HTML 文档。
2. 点击“开始处理”，系统会解析文档并构建或更新本地知识库。
3. 在“检索范围”中选择全部文档，或限定一篇文档。
4. 在右侧输入问题，系统会返回基于文档证据的答案。
5. 点击回答下方的来源按钮，在左侧查看原文并核验页码或高亮片段。
6. 需要保存分析结果时，点击“导出当前对话”下载 Markdown 记录。

## Docker 部署

```bash
docker-compose build
docker-compose up -d
```

启动后按实际配置访问服务端口。

## 评估

项目包含基于 RAGAS 的评估脚本，可用于从答案相关性、上下文召回等维度评估问答质量：

```bash
python eval/evaluate_rag.py
```

## 注意事项

- 首次运行时，BGE-M3 和 Reranker 模型可能需要下载，耗时取决于网络和硬件环境。
- GPU 可提升嵌入和重排序速度；没有 GPU 时系统可使用 CPU 运行，但响应会较慢。
- 问答质量依赖于文档解析质量、知识库内容和模型服务配置。
- 对重要财务结论，请始终通过来源链接和原文进行人工复核。

## License

This project is licensed under the [MIT License](LICENSE).
