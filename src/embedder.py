"""
向量化

1. 支持从 data/processed 读取 txt 和 json（带页码）并自动切片
2. 使用更强大的 BGE-M3 模型
3. 使用 LangChain 接口，方便后续步骤
4. 保留页码元数据用于溯源
"""

import os
import json
import shutil
from pathlib import Path
from typing import List
from tqdm import tqdm

# 引入 LangChain 组件
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

def select_best_gpu():
    """
    选择显存最充足的 GPU
    返回: (device_id, free_memory_gb) 或 (None, 0) 如果没有可用 GPU
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None, 0
        
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            return None, 0
        
        best_gpu = 0
        max_free_memory = 0
        
        print(f"检测到 {num_gpus} 张 GPU，正在检查显存...")
        
        for i in range(num_gpus):
            torch.cuda.set_device(i)
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1e9  # GB
            
            # 获取当前显存使用情况
            allocated = torch.cuda.memory_allocated(i) / 1e9  # GB
            reserved = torch.cuda.memory_reserved(i) / 1e9  # GB
            free_memory = total_memory - reserved  # 可用显存
            
            print(f"   GPU {i}: {props.name}")
            print(f"      总显存: {total_memory:.2f} GB")
            print(f"      已分配: {allocated:.2f} GB")
            print(f"      已保留: {reserved:.2f} GB")
            print(f"      可用显存: {free_memory:.2f} GB")
            
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = i
        
        print(f"选择 GPU {best_gpu} (可用显存: {max_free_memory:.2f} GB)")
        return best_gpu, max_free_memory
        
    except Exception as e:
        print(f"警告: GPU 检测失败: {e}")
        return None, 0

class FinancialVectorDB:
    def __init__(self, 
                 input_dir="data/processed", 
                 persist_dir="data/vector_db",
                 model_name="BAAI/bge-m3"):
        
        self.input_dir = Path(input_dir)
        self.persist_dir = Path(persist_dir)
        
        print(f"正在加载 Embedding 模型: {model_name} ...")
        print("提示: BGE-M3 模型较大，首次运行下载可能需要几分钟，请耐心等待。")
        
        import torch
        
        # 选择最佳 GPU
        best_gpu_id, free_memory = select_best_gpu()
        
        try:
            if best_gpu_id is not None and free_memory > 1.0:  # 至少需要 1GB 可用显存
                device_str = f'cuda:{best_gpu_id}'
                print(f"   使用 GPU {best_gpu_id} 加载模型...")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={
                        'device': device_str,
                        'trust_remote_code': True
                    }, 
                    encode_kwargs={'normalize_embeddings': True}
                )
                print(f"模型加载完毕 (GPU {best_gpu_id} 模式)")
            else:
                raise RuntimeError(f"GPU 不可用或显存不足 (可用: {free_memory:.2f} GB)")
        except Exception as e:
            print(f"警告: GPU 加载失败，切换到 CPU 模式: {e}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': True
                }, 
                encode_kwargs={'normalize_embeddings': True}
            )
            print("模型加载完毕 (CPU 模式)")

    def load_documents(self) -> List[Document]:
        """读取清洗后的 txt 和 json 文件"""
        documents = []
        
        # 读取 txt 文件
        txt_files = list(self.input_dir.glob("*.txt"))
        # 读取 json 文件（带页码元数据）
        json_files = list(self.input_dir.glob("*.json"))
        
        if not txt_files and not json_files:
            print(f"错误: 在 {self.input_dir} 未找到文件。请先运行步骤 2。")
            return []

        print(f"正在读取 {len(txt_files)} 个 TXT 文档和 {len(json_files)} 个 JSON 文档...")
        
        # 处理 TXT 文件
        for file_path in tqdm(txt_files, desc="Loading TXT"):
            try:
                loader = TextLoader(str(file_path), encoding='utf-8')
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file_path.name.replace('.txt', '.pdf')
                documents.extend(docs)
            except Exception as e:
                print(f"警告: 读取失败 {file_path.name}: {e}")
        
        # 处理 JSON 文件（带页码）
        for file_path in tqdm(json_files, desc="Loading JSON"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    source_name = data.get('source', file_path.name)
                    pages = data.get('pages', [])
                    
                    for page_data in pages:
                        doc = Document(
                            page_content=page_data['text'],
                            metadata={
                                "source": source_name,
                                "page": page_data['page']
                            }
                        )
                        documents.append(doc)
            except Exception as e:
                print(f"警告: 读取失败 {file_path.name}: {e}")
        
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """关键步骤：文本切片（保留页码元数据）"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(documents)
        print(f"文档切分完成: 原文档 {len(documents)} -> 切片后 {len(split_docs)} 个片段")
        return split_docs

    def build_db(self, clear_existing=True, incremental=False):
        """
        构建数据库
        Args:
            clear_existing: 是否清空现有数据库
            incremental: 是否增量更新（只添加新文档）
        """
        if clear_existing and self.persist_dir.exists():
            print(f"清理旧数据库: {self.persist_dir}")
            shutil.rmtree(self.persist_dir)

        # 1. 加载
        raw_docs = self.load_documents()
        if not raw_docs: 
            return

        # 2. 切片
        chunks = self.split_documents(raw_docs)

        # 3. 向量化入库（优化批量处理）
        print("正在生成向量并存入数据库...")
        print(f"总共需要处理 {len(chunks)} 个文档片段")
        print("提示：向量化过程可能需要几分钟，请耐心等待...")
        
        if incremental and self.persist_dir.exists():
            # 增量更新：加载现有数据库，只添加新文档
            try:
                vector_store = Chroma(
                    persist_directory=str(self.persist_dir),
                    embedding_function=self.embeddings
                )
                # 获取已存在的文档源，避免重复添加
                existing_sources = set()
                try:
                    existing_docs = vector_store.get()
                    if existing_docs and 'metadatas' in existing_docs:
                        existing_sources = {m.get('source', '') for m in existing_docs['metadatas'] if m}
                except:
                    pass
                
                # 过滤掉已存在的文档（基于源文件名）
                new_chunks = [
                    chunk for chunk in chunks 
                    if chunk.metadata.get('source', '') not in existing_sources
                ]
                
                if new_chunks:
                    print(f"发现 {len(new_chunks)} 个新文档片段，正在批量添加...")
                    # 批量添加，提高性能 - 减小批次大小并添加进度反馈
                    batch_size = 50  # 减小批次大小，避免卡住
                    total_batches = (len(new_chunks) + batch_size - 1) // batch_size
                    
                    for i in range(0, len(new_chunks), batch_size):
                        batch = new_chunks[i:i + batch_size]
                        batch_num = i // batch_size + 1
                        
                        print(f"   正在处理批次 {batch_num}/{total_batches} ({len(batch)} 个片段)...")
                        vector_store.add_documents(batch)
                        print(f"   ✓ 批次 {batch_num} 完成 ({min(i + batch_size, len(new_chunks))}/{len(new_chunks)})")
                else:
                    print("没有新文档需要添加")
            except Exception as e:
                print(f"警告: 增量更新失败，将进行完整重建: {e}")
                # 如果增量更新失败，回退到完整重建（使用批量处理）
                print(f"准备向量化 {len(chunks)} 个文档片段...")
                batch_size = 50
                vector_store = None
                
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    total_batches = (len(chunks) + batch_size - 1) // batch_size
                    
                    print(f"   正在处理批次 {batch_num}/{total_batches} ({len(batch)} 个片段)...")
                    
                    if vector_store is None:
                        vector_store = Chroma.from_documents(
                            documents=batch,
                            embedding=self.embeddings,
                            persist_directory=str(self.persist_dir)
                        )
                    else:
                        vector_store.add_documents(batch)
                    
                    print(f"   ✓ 批次 {batch_num} 完成")
        else:
            # 全新构建或完全重建 - 使用批量处理优化性能
            print(f"准备向量化 {len(chunks)} 个文档片段...")

            # 优化：分批处理，避免一次性处理太多导致卡住
            # 同时提供进度反馈
            batch_size = 50  # 减小批次大小，提高响应性
            vector_store = None

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(chunks) + batch_size - 1) // batch_size

                print(f"   正在处理批次 {batch_num}/{total_batches} ({len(batch)} 个片段)...")

                if vector_store is None:
                    # 第一批：创建新的数据库
                    vector_store = Chroma.from_documents(
                        documents=batch,
                        embedding=self.embeddings,
                        persist_directory=str(self.persist_dir),
                    )
                else:
                    # 后续批次：添加到现有数据库
                    vector_store.add_documents(batch)

                print(f"   ✓ 批次 {batch_num} 完成")
        
        # 确保返回正确的 vector_store 实例
        if incremental and self.persist_dir.exists():
            # 增量更新后，重新加载以确保数据一致
            vector_store = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.embeddings
            )
        
        print(f"数据库构建完成！存储路径: {self.persist_dir}")
        try:
            count = vector_store._collection.count()
            print(f"当前数据库包含 {count} 条向量数据")
        except:
            pass

    def test_retrieval(self, query: str):
        """测试检索效果"""
        print(f"\n测试检索: '{query}'")
        vector_store = Chroma(
            persist_directory=str(self.persist_dir), 
            embedding_function=self.embeddings
        )
        results = vector_store.similarity_search(query, k=3)
        
        for i, doc in enumerate(results):
            print(f"\n[结果 {i+1}] (来源: {doc.metadata.get('source')}, 页码: {doc.metadata.get('page', 'N/A')})")
            print(f"内容: {doc.page_content[:150]}...")

if __name__ == "__main__":
    db = FinancialVectorDB()
    db.build_db()
    db.test_retrieval("贵州茅台的净利润")
