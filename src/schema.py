import uuid
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator

# ==========================================
# 1. 枚举类型定义 (Enums)
# ==========================================

class ChunkType(str, Enum):
    """定义内容块的类型，决定前端如何渲染以及模型如何处理"""
    TEXT = "text"             # 普通文本
    TABLE = "table"           # 表格 (通常是 Markdown 或 HTML 格式)
    IMAGE = "image"           # 图片/图表 (需要配合 image_path)
    SECTION_HEADER = "header" # 标题 (用于构建文档结构树)

# ==========================================
# 2. 核心数据模型 
# ==========================================

class DocumentChunk(BaseModel):
    """
    RAG 系统中流转的最小数据单元。
    
    数据流向: 
    PDF解析(本科生) -> 向量库(研究生) -> 检索(研究生) -> 前端展示(本科生)
    """
    
    # --- 基础标识 ---
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # --- 核心内容 ---
    content: str = Field(..., description="用于向量化和LLM输入的清洗后文本")
    chunk_type: ChunkType = Field(default=ChunkType.TEXT, description="块类型")
    
    # --- 溯源信息 (Frontend 需要) ---
    source_name: str = Field(..., description="来源文件名, 如: 茅台2023年报.pdf")
    page_number: int = Field(..., description="页码 (从1开始)")
    
    # --- 高亮与多模态 (Frontend & Vision Model 需要) ---
    # 格式: [x0, y0, x1, y1] (左上角, 右下角). 若无高亮则为 None
    bbox: Optional[List[float]] = Field(default=None, description="PDF原始坐标，用于前端画红框")
    
    # 原始文本 (用于前端精确匹配高亮，防止content清洗后与原文不一致)
    original_text: Optional[str] = None
    
    # 图片路径 (仅当 chunk_type=IMAGE 时有效)
    image_path: Optional[str] = Field(default=None, description="图片的本地存储路径或URL")

    # --- 检索元数据 (Algo 需要) ---
    score: Optional[float] = Field(default=0.0, description="检索相似度得分/Rerank得分")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外信息，如 parent_section: 'Item 7'")

    # --- 验证逻辑 ---
    @field_validator('bbox')
    def validate_bbox(cls, v):
        if v is not None:
            if len(v) != 4:
                raise ValueError('bbox 必须包含 4 个坐标值 [x0, y0, x1, y1]')
        return v

    def to_chroma_payload(self) -> Dict[str, Any]:
        """
        转换为存入 ChromaDB 的格式。
        Chroma 的 metadata 只能存 str, int, float, bool，不能存 list/dict。
        """
        base_meta = {
            "source": self.source_name,
            "page": self.page_number,
            "type": self.chunk_type.value,
            "image_path": self.image_path or "",
            "bbox": str(self.bbox) if self.bbox else "", # 序列化 list
            "original_text": self.original_text or ""
        }
        # 合并自定义 metadata
        base_meta.update(self.metadata)
        return base_meta

# ==========================================
# 3. 交互模型 
# ==========================================

class UserQuery(BaseModel):
    """用户输入"""
    query: str
    top_k: int = 5
    # 是否开启混合检索
    enable_hybrid: bool = True 

class RetrievalResult(BaseModel):
    """检索结果集，返回给前端"""
    query: str
    results: List[DocumentChunk]
    # 可选: 用于调试的额外信息
    debug_info: Optional[Dict[str, Any]] = None 

# ==========================================
# 4. 辅助工厂方法 
# ==========================================

def create_image_chunk(image_path: str, caption: str, source: str, page: int, bbox: List[float]) -> DocumentChunk:
    """
    快速创建图片块的辅助函数
    """
    return DocumentChunk(
        content=caption, # 图片的描述作为 content 存入向量库
        chunk_type=ChunkType.IMAGE,
        source_name=source,
        page_number=page,
        bbox=bbox,
        image_path=image_path,
        original_text="[IMAGE]" # 占位符
    )