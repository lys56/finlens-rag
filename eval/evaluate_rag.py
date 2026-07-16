import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval import DocumentRetriever
from src.generator import ResponseGenerator


class RAGEvaluator:

    def __init__(self):
        self.retriever = DocumentRetriever()
        self.generator = ResponseGenerator()
    
    def evaluate_with_ragas(self, test_file="eval/test_cases.json"):
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision
            from datasets import Dataset
            
            # 加载测试用例
            with open(test_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            # 准备评测数据
            eval_data = {
                'question': [],
                'answer': [],
                'contexts': [],
                'ground_truth': []
            }
            
            for case in test_data.get("test_cases", []):
                query = case["query"]
                
                # 检索文档
                docs = self.retriever.retrieve_and_rerank(query, top_k=5)
                contexts = [doc['text'] for doc in docs]
                
                # 生成回答
                result = self.generator.generate_with_sources(query, docs)
                
                eval_data['question'].append(query)
                eval_data['answer'].append(result['answer'])
                eval_data['contexts'].append(contexts)
                eval_data['ground_truth'].append(case.get('ground_truth', ''))
            
            # 转换为 Dataset
            dataset = Dataset.from_dict(eval_data)
            
            # 定义评测指标
            metrics = [
                faithfulness,      # 忠实度（防幻觉）
                answer_relevancy,  # 答案相关性
                context_precision  # 上下文精确度
            ]
            
            print("正在进行自动化评测...")
            results = evaluate(dataset=dataset, metrics=metrics)
            
            print("\n评测报告:")
            df_results = results.to_pandas()
            print(df_results[['question', 'faithfulness', 'answer_relevancy', 'context_precision']])
            
            return results
            
        except ImportError:
            print("请安装 ragas: pip install ragas datasets")
            return None
        except Exception as e:
            print(f"评测失败: {e}")
            return None
    
    def simple_evaluation(self):
        """简单评测（不依赖 Ragas）"""
        test_queries = [
            "公司的营收情况如何？",
            "主要的风险因素有哪些？",
            "研发投入占比是多少？"
        ]
        
        print("=" * 50)
        print("简单评测模式")
        print("=" * 50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n[测试 {i}] 查询: {query}")
            print("-" * 50)
            
            # 检索
            docs = self.retriever.retrieve_and_rerank(query, top_k=3)
            print(f"检索到 {len(docs)} 个文档")
            
            # 生成
            result = self.generator.generate_with_sources(query, docs)
            print(f"回答: {result['answer'][:200]}...")
            print()


if __name__ == "__main__":
    evaluator = RAGEvaluator()
    
    # 创建示例测试用例
    test_cases = {
        "test_cases": [
            {
                "query": "公司2023年的营收情况",
                "ground_truth": "营收为3833亿美元，同比下降3%"
            },
            {
                "query": "服务业务的表现如何",
                "ground_truth": "服务业务营收达到852亿美元，创历史新高"
            }
        ]
    }
    
    # 保存测试用例
    test_file = Path(__file__).parent / "test_cases.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)
    
    print("示例测试用例已创建")
    
    # 运行简单评测
    evaluator.simple_evaluation()
    
    print("\n提示: 安装 ragas 后可使用完整的自动化评测:")
    print("   pip install ragas datasets")
