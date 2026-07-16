"""
数据解析与清洗模块
功能：
1. 遍历 my_rag_data 目录下的所有文件
2. 解析 PDF (使用 pdfplumber 提取文本)
3. 解析 HTML (使用 BeautifulSoup 去噪)
4. 将清洗后的文本保存到 data/processed 目录
"""

import os
import re
import pdfplumber
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm  # 进度条库

class FinancialDataProcessor:
    def __init__(self, input_dir="my_rag_data", output_dir="data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def clean_text(self, text: str) -> str:
        """
        通用文本清洗函数
        去除多余空白、乱码、页码等
        """
        if not text:
            return ""
            
        # 1. 替换连续的空白字符（换行、制表符）为单个空格或换行
        # 这里保留换行符，因为金融文档的段落结构很重要
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text) # 保留段落间距
        
        # 2. 去除常见干扰项 (简单的页码匹配，如 "- 1 -", "Page 1 of 50")
        text = re.sub(r'^\s*[-_]?\s*\d+\s*[-_]?\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        
        return text.strip()

    def parse_pdf(self, file_path: Path) -> str:
        """解析 PDF 文件"""
        full_text = []
        try:
            with pdfplumber.open(file_path) as pdf:
                # 遍历每一页
                for page in pdf.pages:
                    # 提取文本
                    text = page.extract_text()
                    if text:
                        full_text.append(text)
                        
            return "\n".join(full_text)
        except Exception as e:
            print(f"警告: PDF 解析失败 {file_path.name}: {e}")
            return ""

    def parse_html(self, file_path: Path) -> str:
        """解析 HTML 文件"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f, 'html.parser')
                
                # 移除 script 和 style 标签
                for script in soup(["script", "style", "head", "title", "meta"]):
                    script.extract()
                
                # 获取文本，使用换行符分隔块级元素
                text = soup.get_text(separator='\n')
                return text
        except Exception as e:
            print(f"警告: HTML 解析失败 {file_path.name}: {e}")
            return ""

    def process_all(self):
        """主处理流程"""
        print(f"🚀 开始处理数据...")
        print(f"📂 输入目录: {self.input_dir}")
        print(f"📂 输出目录: {self.output_dir}")
        
        # 递归查找所有文件
        all_files = list(self.input_dir.rglob("*"))
        # 过滤出文件（排除文件夹）
        files_to_process = [f for f in all_files if f.is_file()]
        
        success_count = 0
        
        # 使用 tqdm 显示进度条
        for file_path in tqdm(files_to_process, desc="Processing"):
            content = ""
            file_ext = file_path.suffix.lower()
            
            # 1. 根据后缀选择解析器
            if file_ext == '.pdf':
                content = self.parse_pdf(file_path)
            elif file_ext in ['.html', '.htm', '.txt']:
                # .txt 也通常包含 html 标签 (SEC 格式)，统一用 html 解析器处理更安全
                content = self.parse_html(file_path)
            else:
                continue # 跳过其他文件
            
            # 2. 清洗文本
            cleaned_content = self.clean_text(content)
            
            if not cleaned_content:
                continue
                
            # 3. 保存结果
            # 保持原始文件名的基础上，加 .txt 后缀
            # 例如: 000001_年报.pdf -> 000001_年报.txt
            output_filename = f"{file_path.stem}.txt"
            output_path = self.output_dir / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
                
            success_count += 1
            
        print("-" * 50)
        print(f" +处理完成! 成功转换 {success_count} 个文件。")
        print(f" 查看结果: {self.output_dir}")

if __name__ == "__main__":
    processor = FinancialDataProcessor(
        input_dir="my_rag_data",      # 对应步骤 1 的下载目录
        output_dir="data/processed"   # 清洗后的存放目录
    )
    processor.process_all()