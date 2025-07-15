import os
import gradio as gr
from dataclasses import dataclass
from typing import List, Optional, Literal
from PIL import Image
import torch
import random
from argparse import ArgumentParser
import pathlib
# FAISSVectorStore：向量数据库，用于存储和检索服装向量
from .vectorstore import FAISSVectorStore
from ..models.load import load_model
from ..data import datatypes
from ..data.datasets import polyvore_utils as polyvore

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
LOGS_DIR = SRC_DIR / 'logs'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(LOGS_DIR, exist_ok=True)

POLYVORE_PRECOMPUTED_REC_EMBEDDING_DIR = "{polyvore_dir}/precomputed_rec_embeddings"
# 设置每页显示12个商品，每次搜索返回8个结果
ITEM_PER_PAGE = 12
ITEM_PER_SEARCH = 8
# 服装类别列表
POLYVORE_CATEGORIES = [
    'bottoms', 'tops',
    'unknown'
]
# 场景标签列表
SCENE_TAGS = ['casual', 'sport']
# 全局状态变量：用户选择的商品和候选商品
state_my_items = []
state_candidate_items = []


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['original', 'clip'],
                        default='clip')
    parser.add_argument('--polyvore_dir', type=str, 
                        default='./src/data/datasets/polyvore')
    parser.add_argument('--checkpoint', type=str, 
                        default=None)
    
    return parser.parse_args()


def run(args):
    
    metadata = polyvore.load_metadata(
        args.polyvore_dir
    )
    items = polyvore.PolyvoreItemDataset(
        args.polyvore_dir, metadata=metadata, load_image=True
    )

    num_pages = len(items) // ITEM_PER_PAGE
    # 根据页码返回对应页的商品列表
    def get_items(page):
        idxs = range(page * ITEM_PER_PAGE, (page + 1) * ITEM_PER_PAGE)
        return [items[i] for i in idxs]

    # 根据场景筛选商品
    def filter_items_by_scene(scene_filter):
        if not scene_filter:
            return list(range(len(items)))
        
        filtered_indices = []
        for i in range(len(items)):
            item = items.get_item_by_id(i)
            if item and hasattr(item, 'scene') and scene_filter in item.scene:
                filtered_indices.append(i)
        return filtered_indices

    # 创建筛选后的FAISS索引
    def create_filtered_index(filtered_indices):
        if not filtered_indices:
            return None
        
        try:
            # 从原始索引中提取筛选后的embedding
            faiss_dir = POLYVORE_PRECOMPUTED_REC_EMBEDDING_DIR.format(polyvore_dir=args.polyvore_dir)
            
            # 加载所有预计算的embedding
            import pickle
            import numpy as np
            
            all_embeddings = []
            all_ids = []
            
            # 读取所有embedding文件
            embedding_files = [f for f in os.listdir(faiss_dir) if f.endswith('.pkl')]
            for filename in sorted(embedding_files):
                filepath = os.path.join(faiss_dir, filename)
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    all_ids.extend(data['ids'])
                    all_embeddings.append(data['embeddings'])
            
            # 合并所有embedding
            all_embeddings = np.concatenate(all_embeddings, axis=0)
            
            # 创建ID到索引的映射
            id_to_idx = {item_id: idx for idx, item_id in enumerate(all_ids)}
            
            # 提取筛选后的embedding
            filtered_embeddings = []
            filtered_ids = []
            
            for item_id in filtered_indices:
                if item_id in id_to_idx:
                    idx = id_to_idx[item_id]
                    filtered_embeddings.append(all_embeddings[idx])
                    filtered_ids.append(item_id)
            
            if not filtered_embeddings:
                return None
            
            # 创建临时FAISS索引
            filtered_embeddings = np.array(filtered_embeddings)
            temp_indexer = FAISSVectorStore(
                index_name='temp_filtered_index',
                d_embed=128,
                faiss_type='IndexFlatIP',
                base_dir="",  # 使用空字符串
            )
            
            # 添加筛选后的embedding到临时索引
            temp_indexer.add(embeddings=filtered_embeddings.tolist(), ids=filtered_ids)
            
            return temp_indexer
            
        except Exception as e:
            print(f"[WARNING] 创建筛选索引失败: {e}")
            return None

    
    model = load_model(
        model_type=args.model_type, checkpoint=args.checkpoint
    )
    model.eval()
    indexer = FAISSVectorStore(
        index_name='rec_index',
        d_embed=128,
        # IndexFlatIP：使用内积相似度进行检索
        faiss_type='IndexFlatIP',
        base_dir=POLYVORE_PRECOMPUTED_REC_EMBEDDING_DIR.format(polyvore_dir=args.polyvore_dir),
    )
    '''-----------------------------Web界面构建-----------------------------'''
    # 创建Gradio Web界面
    with gr.Blocks() as demo:
        # 存储当前选中的商品索引
        state_selected_my_item_index = gr.State(value=None)
        
        with gr.Row(equal_height=True):
            gr.Markdown(
                "# Outfit Recommendation Demo"
            )
        
        with gr.Row(equal_height=True):
            gr.Markdown(
                "## My Items"
            )
        with gr.Row(equal_height=True):
            with gr.Column(scale=10, variant='compact'):
                with gr.Row(equal_height=True):
                    with gr.Column(variant='compact'):
                        my_item_gallery = gr.Gallery(
                            allow_preview=False, show_label=True,
                            columns=4, rows=1,
                        )       
            with gr.Column(scale=2, variant='compact'):
                # 商品类别选择
                with gr.Row(equal_height=True):
                    with gr.Column(variant='compact'):
                        item_category = gr.Dropdown(
                            label="Category",
                            choices=POLYVORE_CATEGORIES, value=None,
                        )
                with gr.Row(equal_height=True):
                    with gr.Column(variant='compact'):
                        item_image = gr.Image(
                            label="Upload Image", type="pil",
                        )
                with gr.Row(equal_height=True):
                    with gr.Column(variant='compact'):
                        item_description = gr.Textbox(
                            label="Description",
                        )
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=100, variant='compact'):
                        btn_item_add = gr.Button("Add")
                    with gr.Column(scale=1, min_width=100, variant='compact'):
                        btn_item_delete = gr.Button("Delete")
        
        # 从Polyvore添加商品
        '''显示Polyvore数据集中的商品
            分页显示，每页12个商品
            用户可以从这里选择商品添加到自己的衣柜'''
        with gr.Row(equal_height=True):
            gr.Markdown(
                "## Add From Polyvore"
            )
        with gr.Row(equal_height=True):
            with gr.Column(scale=12, variant='compact'):
                with gr.Row(equal_height=True):
                    with gr.Column(variant='compact'):
                        polyvore_gallery = gr.Gallery(
                            allow_preview=False,
                            show_label=True,
                            columns=ITEM_PER_PAGE // 2, 
                            rows=2,
                            type="pil",
                            object_fit='contain',
                            height='auto'
                        )
                with gr.Row(equal_height=True):
                    with gr.Column(variant='compact'):
                        polyvore_page = gr.Dropdown(
                            label="Page",
                            choices=list(range(1, num_pages + 1)),  # 1부터 num_pages까지 선택 가능
                            value=None  # 기본값
                        )
        
        
        with gr.Row(equal_height=True):
            gr.Markdown(
                "## Task"
            )
        with gr.Row(equal_height=True, variant='compact'):
            with gr.Column(scale=1, variant='compact'):
                with gr.Row(equal_height=True):
                    # 计算兼容性分数
                    gr.Markdown(
                        "Compute Score"
                    )
                with gr.Row(equal_height=True, variant='compact'):
                    btn_compute_score = gr.Button(
                        "Compute",
                        variant="primary"
                    )
                with gr.Row(equal_height=True, variant='compact'):
                    computed_score = gr.Textbox(
                        label="Compatibility Score",
                        interactive=False
                    )
                        
            with gr.Column(scale=2, variant='compact'):
                # 搜索互补商品
                with gr.Row(equal_height=True):
                    gr.Markdown(
                        "Search Complementary Items"
                    )
                # 新增：目标单品描述和场合输入
                with gr.Row(equal_height=True, variant='compact'):
                    comp_description = gr.Textbox(
                        label="Target Item Description (Optional)",
                        placeholder="如：黑色牛仔裤，适合休闲场合",
                    )
                with gr.Row(equal_height=True, variant='compact'):
                    comp_scene = gr.Dropdown(
                        label="Target Item Scene (Optional)",
                        choices=[''] + SCENE_TAGS, value='',
                    )
                with gr.Row(equal_height=True, variant='compact'):
                    btn_search_item = gr.Button(
                        "Search", variant="primary"
                    )
                with gr.Row(equal_height=True, variant='compact'):
                    searched_item_gallery = gr.Gallery(
                        allow_preview=False,
                        show_label=True,
                        columns=ITEM_PER_SEARCH // 2, 
                        rows=2,
                        type="pil",
                        object_fit='contain',
                        height='auto'
                    )
              
        # Functions
        # 商品选择函数 ：更新选中的商品索引，并显示选中的商品信息
        def select_item(selected: gr.SelectData):

            return {
                state_selected_my_item_index: selected.index,
                item_image: state_my_items[selected.index].image,
                item_description: state_my_items[selected.index].description,
                item_category: state_my_items[selected.index].category,
            }
        # 添加商品函数：将新商品添加到用户衣柜
        def add_item(item_image, item_description, item_category):
            global state_my_items
            
            if item_image is None or item_description is None or item_category is None:
                gr.Warning("Error: All fields (image, description, and category) must be provided.")
            else:
                state_my_items.append(
                    datatypes.FashionItem(
                        item_id=None,
                        image=item_image, 
                        description=item_description,
                        category=item_category,
                    )
                )
                
            return {
                my_item_gallery: [item.image for item in state_my_items],
                state_selected_my_item_index: None,
            }
        # 删除商品函数：从用户衣柜中删除指定索引的商品
        def delete_item(index):
            if index is not None:
                if index < len(state_my_items):
                    del state_my_items[index]
                else:
                    gr.Warning("Error: Invalid item index.")
            else:
                gr.Warning("Error: No item selected.")
            
            return {
                my_item_gallery: [item.image for item in state_my_items],
                state_selected_my_item_index: None,
                item_image: None,
                item_description: None,
                item_category: None,
            }
                
        def select_page_from_polyvore(page):
            global state_candidate_items
            
            page = page - 1
            state_candidate_items = get_items(page)
            
            return {
                polyvore_gallery: [item.image for item in state_candidate_items],
            }
        
        def select_item_from_polyvore(selected: gr.SelectData):
            selected_item = state_candidate_items[selected.index]
            
            return {
                item_image: selected_item.image,
                item_description: selected_item.description,
                item_category: selected_item.category,
            }
        # 计算兼容性分数函数：计算用户衣柜中商品的兼容性分数
        @torch.no_grad()
        def compute_score():
            if len(state_my_items) == 0:
                gr.Warning("Error: No items to compute score.")
                return {
                    computed_score: None
                }
            query = datatypes.FashionCompatibilityQuery(
                outfit=state_my_items
            )
            s = model.predict_score(
                query= [query],
                use_precomputed_embedding=False
            )[0].detach().cpu()
            s = float(s)
            
            return {
                computed_score: s
            }
        # 搜索互补商品函数：根据用户衣柜中的商品，搜索与之互补的商品
        @torch.no_grad()
        def search_item(comp_description, comp_scene):
            # 1. 必须选择一个数据库中的单品
            if len(state_my_items) == 0:
                gr.Warning("Error: 请先从数据库选择一个上装或下装！")
                return {
                    searched_item_gallery: []
                }
            user_item = state_my_items[0]  # 只支持单选
            if user_item.category not in ['tops', 'bottoms']:
                gr.Warning("Error: 请选择上装或下装！")
                return {
                    searched_item_gallery: []
                }
            # 2. 自动推断互补类别
            if user_item.category == 'tops':
                target_category = 'bottoms'
            else:
                target_category = 'tops'
            # 3. 先按互补类别筛选
            candidate_indices = [i for i in range(len(items)) if getattr(items.get_item_by_id(i), 'category', None) == target_category]
            # 4. 如有场合再筛选
            if comp_scene:
                candidate_indices = [i for i in candidate_indices if comp_scene in getattr(items.get_item_by_id(i), 'scene', [])]
            if not candidate_indices:
                gr.Warning("Error: 没有符合条件的互补单品！")
                return {
                    searched_item_gallery: []
                }
            # 5. 获取用户单品图像embedding
            user_img_emb = model.image_encoder(
                user_item.image.unsqueeze(0) if hasattr(user_item.image, 'unsqueeze') else user_item.image
            )
            # 6. 如有description，编码文本embedding
            if comp_description:
                text_emb = model.text_encoder([comp_description])
                # 7. 拼接图像和文本embedding
                fusion_emb = torch.cat([user_img_emb, text_emb], dim=-1)
            else:
                fusion_emb = user_img_emb
            # 8. 送入transformer生成检索embedding
            query_emb = model.transformer(fusion_emb).detach().cpu().numpy().tolist()
            # 9. 创建筛选后的FAISS索引
            filtered_indexer = create_filtered_index(candidate_indices)
            if filtered_indexer is None:
                filtered_indexer = indexer
            # 10. 检索
            res = filtered_indexer.search(
                embeddings=query_emb,
                k=min(ITEM_PER_SEARCH, len(candidate_indices))
            )[0]
            # 11. 返回图片
            return {
                searched_item_gallery: [items.get_item_by_id(r[1]).image for r in res]
            }
        
        
        # Event Handlers
        my_item_gallery.select(
            select_item,
            inputs=None,
            outputs=[state_selected_my_item_index, item_image, item_description, item_category]
        )
        btn_item_add.click(
            add_item, 
            inputs=[item_image, item_description, item_category], 
            outputs=[my_item_gallery, state_selected_my_item_index]
        )
        btn_item_delete.click(
            delete_item,
            inputs=state_selected_my_item_index,
            outputs=[my_item_gallery, state_selected_my_item_index, item_image, item_description, item_category]
        )
        
        polyvore_page.change(
            select_page_from_polyvore,
            inputs=[polyvore_page],
            outputs=[polyvore_gallery]
        )
        polyvore_gallery.select(
            select_item_from_polyvore,
            inputs=None,
            outputs=[item_image, item_description, item_category]
        )
        
        btn_compute_score.click(
            compute_score,
            inputs=None,
            outputs=[computed_score]
        )
        btn_search_item.click(
            search_item,
            inputs=[comp_description, comp_scene],
            outputs=[searched_item_gallery]
        )
    
    # Launch
    demo.launch(server_name="0.0.0.0", server_port=7860)    
if __name__ == "__main__":
    args = parse_args()
    run(args)