from typing import List, Optional, TypedDict, Union
from PIL import Image
import copy
from pydantic import BaseModel, Field
import numpy as np

# todo：添加场景分类：运动、日常
class FashionItem(BaseModel):
    item_id: Optional[int] = Field(
        default=None,
        description="Unique ID of the item, mapped to `id` in the ItemLoader"
    )
    # 上下装分类
    category: Optional[str] = Field(
        default="",
        description="Category of the item"
    )
    image: Optional[Image.Image] = Field(
        default=None,
        description="Image of the item"
    )
    description: Optional[str] = Field(
        default="",
        description="Description of the item"
    )
    metadata: Optional[dict] = Field(
        default_factory=dict,
        description="Additional metadata for the item"
    )
    embedding: Optional[np.ndarray] = Field(
        default=None,
        description="Embedding of the item"
    )

    class Config:
        arbitrary_types_allowed = True

# 兼容性查询
class FashionCompatibilityQuery(BaseModel):
    outfit: List[Union[FashionItem, int]] = Field(
        default_factory=list,
        description="List of fashion items"
    )

# 互补性查询
class FashionComplementaryQuery(BaseModel):
    outfit: List[Union[FashionItem, int]] = Field(
        # 其实只有一件
        default_factory=list,
        description="List of fashion items"
    )
    category: str = Field(
        default="",
        description="Category of the target outfit"
    )
       
class FashionCompatibilityData(TypedDict):
    label: Union[
        int, 
        List[int]
    ]
    query: Union[
        FashionCompatibilityQuery, 
        List[FashionCompatibilityQuery]
    ]
    
    
class FashionFillInTheBlankData(TypedDict):
    query: Union[
        FashionComplementaryQuery,
        List[FashionComplementaryQuery]
    ]
    label: Union[
        int,
        List[int]
    ]
    candidates: Union[
        List[FashionItem],
        List[List[FashionItem]]
    ]
    
    
class FashionTripletData(TypedDict):
    query: Union[
        FashionComplementaryQuery,
        List[FashionComplementaryQuery]
    ]
    answer: Union[
        FashionItem,
        List[FashionItem]
    ]