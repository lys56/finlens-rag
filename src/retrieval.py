"""
检索与重排序模块
功能：
1. 使用 LangChain 加载已有的 ChromaDB (BGE-M3)
2. 向量初步检索 (粗排)
3. 使用 BGE-Reranker-v2-m3 进行深度重排序 (精排) - 强制 GPU 加速 + 批处理优化
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# 引入必要的库
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

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

class DocumentRetriever:
    def __init__(self, 
                 db_path="data/vector_db", 
                 embedding_model="BAAI/bge-m3",
                 rerank_model="BAAI/bge-reranker-v2-m3"):
        
        self.db_path = Path(db_path)
        
        # 检查数据库是否存在
        if not self.db_path.exists():
            raise FileNotFoundError(f"向量数据库不存在: {self.db_path}\n请先上传文档并进行处理。")
        
        # 1. 初始化 Embedding 
        import torch
        print("正在加载 Embedding 模型...")
        
        # 选择最佳 GPU
        best_gpu_id, free_memory = select_best_gpu()
        
        try:
            if best_gpu_id is not None and free_memory > 1.0:  # 至少需要 1GB 可用显存
                device_str = f'cuda:{best_gpu_id}'
                print(f"   使用 GPU {best_gpu_id} 加载模型...")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model,
                    model_kwargs={
                        'device': device_str,
                        'trust_remote_code': True
                    },
                    encode_kwargs={'normalize_embeddings': True}
                )
                print(f"   GPU {best_gpu_id} 加载成功")
            else:
                raise RuntimeError(f"GPU 不可用或显存不足 (可用: {free_memory:.2f} GB)")
        except Exception as e:
            print(f"警告: GPU 加载失败: {e}")
            print("   切换到 CPU 模式...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': True
                },
                encode_kwargs={'normalize_embeddings': True}
            )
            print("   CPU 加载成功")
        
        # 2. 加载数据库
        # 注意: 只有在第一次构建时才需要 persist_directory，读取时直接加载即可
        try:
            self.vector_store = Chroma(
                persist_directory=str(self.db_path),
                embedding_function=self.embeddings,
                collection_name="langchain" 
            )
            print("数据库已连接")
        except Exception as e:
            raise RuntimeError(f"数据库加载失败: {e}\n请检查数据库文件是否完整。")

        # 3. 初始化 Reranker
        self.reranker_model_name = rerank_model
        self.reranker = self._init_reranker()

    def _init_reranker(self):
        """
        初始化重排序模型
        注意：如果遇到 meta tensor 错误，可能是 FlagEmbedding 版本问题
        解决方案：pip install --upgrade FlagEmbedding
        """
        try:
            from FlagEmbedding import FlagReranker
            import torch
            import os
            
            print("正在加载重排序模型...")
            
            # 选择最佳 GPU
            best_gpu_id, free_memory = select_best_gpu()
            
            # 设置环境变量，避免 meta tensor 错误
            # 禁用 accelerate 的自动卸载功能
            os.environ.setdefault('ACCELERATE_USE_CPU', '0')
            
            try:
                if best_gpu_id is not None and free_memory > 1.0:  # 至少需要 1GB 可用显存
                    # GPU 模式：使用选中的 GPU
                    device_str = f'cuda:{best_gpu_id}'
                    print(f"   使用 GPU {best_gpu_id} 加载重排序模型...")
                    
                    reranker = FlagReranker(
                        self.reranker_model_name, 
                        use_fp16=True,
                        device=device_str
                    )
                    print(f"   GPU {best_gpu_id} 加载成功")
                    return reranker
                else:
                    raise RuntimeError(f"GPU 不可用或显存不足 (可用: {free_memory:.2f} GB)")
            except RuntimeError as e:
                if "meta tensor" in str(e).lower() or "Cannot copy out of meta" in str(e):
                    print(f"错误: Meta tensor 错误: {e}")
                    print("解决方案：")
                    print("      1. 升级 FlagEmbedding: pip install --upgrade FlagEmbedding")
                    print("      2. 或使用 CPU 模式（自动切换）")
                    # 强制使用 CPU
                    print("   强制切换到 CPU 模式...")
                    reranker = FlagReranker(
                        self.reranker_model_name, 
                        use_fp16=False,
                        device='cpu'
                    )
                    print("   CPU 加载成功")
                    return reranker
                else:
                    raise
            except Exception as e:
                print(f"警告: GPU 加载失败: {e}")
                print("   切换到 CPU 模式...")
                # CPU 模式：不使用 fp16
                reranker = FlagReranker(
                    self.reranker_model_name, 
                    use_fp16=False,
                    device='cpu'
                )
                print("   CPU 加载成功")
                return reranker
        except Exception as e:
            print(f"警告: 重排序模型加载失败: {e}")
            print("   将使用粗排结果（无重排序）")
            print("提示：重排序功能已禁用，但基本检索功能仍可用")
            return None

    def retrieve_and_rerank(
        self,
        query: str,
        top_k: int = 5,
        retrieve_k: int = 20,
        source_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        双阶段检索：粗排 -> 精排
        """
        print("正在粗排检索...")
        # 粗排：从数据库中快速获取 retrieve_k 个候选
        filter_kwargs = {"filter": {"source": source_filter}} if source_filter else {}
        initial_docs = self.vector_store.similarity_search(
            query, k=retrieve_k, **filter_kwargs
        )
        
        if not initial_docs:
            return []

        # 格式化候选文档
        candidate_docs = []
        for doc in initial_docs:
            candidate_docs.append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })

        if not self.reranker:
            return candidate_docs[:top_k]

        print("正在进行重排序...")
        try:
            # 构建 (Query, Document) 对
            pairs = [[query, doc['text']] for doc in candidate_docs]
            
            # 优化点：添加 batch_size 参数
            # 将所有计算任务分批送入 GPU，避免显存溢出同时最大化并行效率
            # 16 或 32 是比较保守且高效的值
            scores = self.reranker.compute_score(pairs, batch_size=16)

            # --- 兼容性处理：确保 scores 是列表 ---
            if isinstance(scores, (float, int)):
                scores = [scores]
            elif hasattr(scores, "tolist"): # 处理 numpy 数组
                scores = scores.tolist()
            
            # 将分数写回文档对象
            for i, doc in enumerate(candidate_docs):
                doc['rerank_score'] = scores[i]

            # 根据分数降序排列
            reranked_docs = sorted(candidate_docs, key=lambda x: x['rerank_score'], reverse=True)
            
            return reranked_docs[:top_k]

        except Exception as e:
            print(f"错误: 重排序过程出错: {e}")
            # 出错时降级返回粗排结果
            return candidate_docs[:top_k]

if __name__ == "__main__":
    # 测试代码
    retriever = DocumentRetriever(db_path="data/vector_db")
    test_query = "贵州茅台2024年的经营目标和分红政策是什么？"
    
    print("=" * 50)
    print(f"提问: {test_query}")
    print("=" * 50)
    
    results = retriever.retrieve_and_rerank(test_query, top_k=3)
    
    for i, doc in enumerate(results, 1):
        score = doc.get('rerank_score')
        score_display = f"{score:.4f}" if isinstance(score, (float, int)) else "N/A"
        
        print(f"\n[排名 {i}] 分数: {score_display}")
        print(f"来源: {doc['metadata'].get('source')}")
        # 移除换行符以便预览
        print(f"内容预览: {doc['text'][:100].replace(chr(10), ' ')}...")
