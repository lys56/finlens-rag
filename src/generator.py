"""
 LLM 生成模块
功能：
1. 接收重排后的文档片段
2. 构建专业的金融分析 Prompt
3. 引导大模型输出带引用来源的严谨回答
"""

import os
from typing import List, Dict, Optional
from openai import OpenAI

class ResponseGenerator:
    """响应生成器 - 支持金融级防幻觉和精准引用"""
    
    def __init__(self, 
                 model_name="qwen-max", 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        初始化生成器
        Args:
            model_name: 使用的模型名称 (推荐 deepseek-chat, qwen-max 等)
            api_key: API 密钥，优先从环境变量获取
            base_url: API 基础地址
        """
        self.model_name = os.getenv("LLM_MODEL", model_name)
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv(
            "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        self.client = self._init_client()
    
    def _init_client(self):
        """初始化 OpenAI 兼容客户端"""
        if not self.api_key:
            print("警告: 未设置 API Key，生成器将运行在【模拟模式】")
            return None
        try:
            return OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            print(f"客户端初始化失败: {e}")
            return None

    def build_prompt(self, query: str, context_docs: List[Dict]) -> tuple:

        # 组装参考内容
        context_str = ""
        for i, doc in enumerate(context_docs):
            source = doc.get('metadata', {}).get('source', '未知文档')
            content = doc.get('text', '').replace('\n', ' ') # 压缩多余换行
            context_str += f"----- 文档 [{i+1}] (来源: {source}) -----\n{content}\n\n"
        
        # 核心金融分析 Prompt
        system_prompt = """你是一个专业的金融分析师。你的任务是根据提供的【参考文档】回答用户问题。

            必须遵守的【铁律】：
            1. 准确性：仅使用【参考文档】中的数据。如果文档中没有相关财务指标或明确说法，请回答“根据现有文档，无法找到相关信息”。
            2. 引用：在每一个涉及事实、数据、结论的句子后，必须标注其来源文档编号。格式如：“2024年净利润为100亿 [1]”。
            3. 金融规范：保留原始货币单位（如“亿元”、“HKD”）。
            4. 结构化：如果涉及多个维度的分析（如收入、分红、前景），请使用分点列出，保持逻辑清晰。
            5. 禁止猜测：严禁基于自身训练知识回答财报中未提及的事实。"""

        user_prompt = f"""【参考文档内容】：
        {context_str}
        
        【用户问题】：{query}
        
        请基于上述参考文档，用清晰、友好、专业的方式回答用户的问题。确保：
        1. 回答准确、有据可查
        2. 语言通俗易懂
        3. 适当标注来源
        4. 如果信息不足，礼貌地说明"""

        return system_prompt, user_prompt

    def generate(self, query: str, context_docs: List[Dict]) -> str:
        """
        衔接检索结果并生成回答（简化返回，直接返回答案文本）
        Args:
            query: 用户问题
            context_docs: retrieval.py 返回的重排后的文档列表
        Returns:
            答案文本字符串
        """
        if not context_docs:
            return "抱歉，检索阶段未能找到相关参考文档。请尝试使用其他关键词或检查文档是否已正确上传。"

        if not self.client:
            return "[模拟模式] 检索到了相关文档，请配置 API Key 以获取 AI 分析结果。"

        try:
            system_p, user_p = self.build_prompt(query, context_docs)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_p},
                    {"role": "user", "content": user_p}
                ],
                temperature=0.2,  # 稍微提高温度，使回答更自然
                max_tokens=2000  # 增加最大token，支持更详细的回答
            )
            
            answer = response.choices[0].message.content
            return answer

        except Exception as e:
            return f"生成失败: {str(e)}\n\n提示：请检查 API 配置或网络连接。"


if __name__ == "__main__":
    # 模拟 retrieval.py 的输出
    mock_docs = [
        {
            "text": "2024年贵州茅台经营目标：实现营业总收入较上年度增长15%左右。",
            "metadata": {"source": "600519_2024年报.txt"},
            "rerank_score": 0.95
        }
    ]
    
    # 请确保已设置环境变量 LLM_API_KEY
    gen = ResponseGenerator()
    res = gen.generate("茅台2024年的经营目标是什么？", mock_docs)
    
    print("-" * 50)
    print(f"回答：\n{res['answer']}")
