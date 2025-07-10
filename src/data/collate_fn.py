from typing import List

from .datatypes import (
    FashionCompatibilityData,
    FashionFillInTheBlankData,
    FashionTripletData,
    FashionItem
)


def item_collate_fn(batch) -> List[FashionItem]:
    return [item for item in batch]


def cp_collate_fn(batch) -> FashionCompatibilityData:
    # 兼容性查询
    # label兼容性标签：0-不兼容，1-兼容
    label = [item['label'] for item in batch]
    query = [item['query'] for item in batch]
    
    return FashionCompatibilityData(
        label=label,
        query=query
    )
    

def fitb_collate_fn(batch) -> FashionFillInTheBlankData:
    # 互补性查询
    # query上下文物品（如已搭配的上衣和裤子）
    # label互补性标签：0-不互补，1-互补
    query = [item['query'] for item in batch]
    label = [item['label'] for item in batch]
    candidates = [item['candidates'] for item in batch]
    
    return FashionFillInTheBlankData(
        query=query,
        label=label,
        candidates=candidates
    )

# query: 所有样本的锚点物品列表。
# answer: 所有样本的正样本列表。
def triplet_collate_fn(batch) -> FashionTripletData:
    query = [item['query'] for item in batch]
    answer = [item['answer'] for item in batch]
    
    return FashionTripletData(
        query=query,
        answer=answer
    )