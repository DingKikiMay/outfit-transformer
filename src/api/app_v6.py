#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask API服务器 V6
优化版本：通过HTTP API间接访问数据库，简化请求体，只需要ID即可
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from flask import Flask, request, jsonify
from flask_cors import CORS

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.fashion_recommendation_api_v6 import FashionRecommendationAPIV6, UserInput, ProductInfo, RecommendationResult

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局API实例
recommendation_api = None


def init_api():
    """初始化推荐API"""
    global recommendation_api
    
    # 从环境变量获取配置，适配AutoDL环境
    # 使用项目根目录的相对路径，避免硬编码绝对路径
    project_root = Path(__file__).parent.parent.parent
    
    # 默认路径配置（可覆盖）
    model_path = os.getenv('MODEL_PATH', str(project_root / 'models' / 'cir_best_model.pth'))
    model_type = os.getenv('MODEL_TYPE', 'clip')
    faiss_index_path = os.getenv('FAISS_INDEX_PATH', '/root/fashion-ai-project/faiss_index.faiss')
    api_base_url = os.getenv('BACKEND_API_URL', 'https://m1.apifoxmock.com/m1/6328147-0-default')
    ssl_verify = os.getenv('SSL_VERIFY', 'true').lower() == 'true'  # SSL证书验证开关
    
    # 确保目录存在
    os.makedirs(Path(model_path).parent, exist_ok=True)
    os.makedirs(Path(faiss_index_path).parent, exist_ok=True)
    
    logger.info(f"项目根目录: {project_root}")
    logger.info(f"模型路径: {model_path}")
    logger.info(f"FAISS索引路径: {faiss_index_path}")
    logger.info(f"后端API地址: {api_base_url}")
    
    try:
        recommendation_api = FashionRecommendationAPIV6(
            model_path=model_path,
            model_type=model_type,
            faiss_index_path=faiss_index_path,
            api_base_url=api_base_url
        )
        # 设置SSL验证配置
        recommendation_api.ssl_verify = ssl_verify
        logger.info("推荐API V6初始化成功")
        return True
    except Exception as e:
        logger.error(f"推荐API V6初始化失败: {e}")
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    if recommendation_api is None:
        return jsonify({
            'status': 'error',
            'message': 'API未初始化'
        }), 500
    
    return jsonify({
        'status': 'ok',
        'message': 'API运行正常',
        'api_info': recommendation_api.get_api_info()
    })

# 从后端API构建FAISS索引接口（预处理阶段）
@app.route('/api/build_faiss_index_from_api', methods=['POST'])
def build_faiss_index_from_api():
    """从后端API构建FAISS索引接口（预处理阶段）"""
    if recommendation_api is None:
        return jsonify({
            'success': False,
            'message': 'API未初始化',
            'data': None
        }), 500
    
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': '请求数据为空',
                'data': None
            }), 400
        
        # 获取参数
        scene = data.get('scene')  # 可选，如果为None则获取所有场景
        # 使用项目根目录的相对路径
        project_root = Path(__file__).parent.parent.parent
        default_save_path = str(project_root / 'faiss_index.faiss')
        save_path = data.get('save_path', default_save_path)
        
        # 构建FAISS索引
        logger.info(f"开始从API构建FAISS索引，场景: {scene or 'all'}")
        valid_count = recommendation_api.build_faiss_index_from_api(scene, save_path)
        
        return jsonify({
            'success': True,
            'message': '分类FAISS索引构建成功',
            'data': {
                'scene': scene or 'all',
                'valid_products': valid_count,
                'save_path': save_path,
                'tops_index_path': save_path.replace('.faiss', '_tops.faiss'),
                'bottoms_index_path': save_path.replace('.faiss', '_bottoms.faiss'),
                'tops_metadata_path': save_path.replace('.faiss', '_tops_metadata.json'),
                'bottoms_metadata_path': save_path.replace('.faiss', '_bottoms_metadata.json')
            }
        })
        
    except Exception as e:
        logger.error(f"构建FAISS索引过程中出现错误: {e}")
        return jsonify({
            'success': False,
            'message': f'构建FAISS索引失败: {str(e)}',
            'data': None
        }), 500


@app.route('/api/recommend_best_item', methods=['POST'])
def recommend_best_item():
    """推荐最佳单品接口（在线检索阶段，返回top-1）"""
    if recommendation_api is None:
        return jsonify({
            'success': False,
            'message': 'API未初始化',
            'data': None
        }), 500
    
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': '请求数据为空',
                'data': None
            }), 400
        
        # 验证必需字段
        if 'product_id' not in data:
            return jsonify({
                'success': False,
                'message': '缺少必需字段: product_id',
                'data': None
            }), 400
        
        # 构建用户输入（简化版）
        user_input = UserInput(
            product_id=data.get('product_id'),
            scene=data.get('scene')
        )
        
        # 调用推荐API（全库检索，返回top-1最佳单品）
        result = recommendation_api.recommend_best_item(user_input)
        
        if result is None:
            return jsonify({
                'success': False,
                'message': '没有找到合适的推荐单品',
                'data': None
            }), 404
        
        # 转换结果格式
        result_data = {
            'product_id': result.product_id
        }
        
        return jsonify({
            'success': True,
            'message': '推荐成功',
            'data': result_data
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'message': str(e),
            'data': None
        }), 400
    except Exception as e:
        logger.error(f"推荐过程中出现错误: {e}")
        return jsonify({
            'success': False,
            'message': f'推荐失败: {str(e)}',
            'data': None
        }), 500


@app.route('/api_info', methods=['GET'])
def get_api_info():
    """获取API信息"""
    if recommendation_api is None:
        return jsonify({
            'success': False,
            'message': 'API未初始化',
            'data': None
        }), 500
    
    return jsonify({
        'success': True,
        'message': '获取成功',
        'data': recommendation_api.get_api_info()
    })


@app.route('/test', methods=['POST'])
def test_recommendation():
    """测试推荐接口"""
    if recommendation_api is None:
        return jsonify({
            'success': False,
            'message': 'API未初始化',
            'data': None
        }), 500
    
    try:
        # 示例测试数据（简化版）
        test_data = {
            'product_id': 1,
            'scene': 'casual'
        }
        
        # 构建用户输入
        user_input = UserInput(
            product_id=test_data['product_id'],
            scene=test_data['scene']
        )
        
        # 调用推荐API（全库检索）
        result = recommendation_api.recommend_best_item(user_input)
        
        if result is None:
            return jsonify({
                'success': False,
                'message': '测试失败：没有找到合适的推荐单品',
                'data': None
            }), 404
        
        # 转换结果格式
        result_data = {
            'product_id': result.product_id
        }
        
        return jsonify({
            'success': True,
            'message': '测试成功',
            'data': result_data
        })
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        return jsonify({
            'success': False,
            'message': f'测试失败: {str(e)}',
            'data': None
        }), 500


@app.route('/api/test_subcategory', methods=['GET'])
def test_subcategory():
    """测试获取子类别接口"""
    if recommendation_api is None:
        return jsonify({
            'success': False,
            'message': 'API未初始化',
            'data': None
        }), 500
    
    try:
        parent_id = request.args.get('parentId', type=int)
        if parent_id is None:
            return jsonify({
                'success': False,
                'message': '缺少parentId参数',
                'data': None
            }), 400
        
        subcategory_ids = recommendation_api._get_subcategory_ids(parent_id)
        
        return jsonify({
            'success': True,
            'message': '获取子类别成功',
            'data': {
                'parent_id': parent_id,
                'subcategory_ids': subcategory_ids,
                'count': len(subcategory_ids)
            }
        })
        
    except Exception as e:
        logger.error(f"测试子类别接口出现错误: {e}")
        return jsonify({
            'success': False,
            'message': f'测试失败: {str(e)}',
            'data': None
        }), 500


@app.route('/api/test_products', methods=['GET'])
def test_products():
    """测试获取商品接口"""
    if recommendation_api is None:
        return jsonify({
            'success': False,
            'message': 'API未初始化',
            'data': None
        }), 500
    
    try:
        type_id = request.args.get('typeId', type=int)
        if type_id is None:
            return jsonify({
                'success': False,
                'message': '缺少typeId参数',
                'data': None
            }), 400
        
        products = recommendation_api._get_products_by_type_id(type_id, page_size=10)
        
        return jsonify({
            'success': True,
            'message': '获取商品成功',
            'data': {
                'type_id': type_id,
                'products': products,
                'count': len(products)
            }
        })
        
    except Exception as e:
        logger.error(f"测试商品接口出现错误: {e}")
        return jsonify({
            'success': False,
            'message': f'测试失败: {str(e)}',
            'data': None
        }), 500


@app.route('/api/test_category_index', methods=['GET'])
def test_category_index():
    """测试分类索引接口"""
    if recommendation_api is None:
        return jsonify({
            'success': False,
            'message': 'API未初始化',
            'data': None
        }), 500
    
    try:
        product_id = request.args.get('productId', type=int)
        if product_id is None:
            # 返回分类索引统计信息
            return jsonify({
                'success': True,
                'message': '获取分类索引统计成功',
                'data': {
                    'total_products': len(recommendation_api.product_category_index),
                    'tops_count': sum(1 for cat in recommendation_api.product_category_index.values() if cat == 'tops'),
                    'bottoms_count': sum(1 for cat in recommendation_api.product_category_index.values() if cat == 'bottoms')
                }
            })
        else:
            # 查询特定商品的分类
            if product_id in recommendation_api.product_category_index:
                category = recommendation_api.product_category_index[product_id]
                return jsonify({
                    'success': True,
                    'message': '获取商品分类成功',
                    'data': {
                        'product_id': product_id,
                        'category': category
                    }
                })
            else:
                return jsonify({
                    'success': False,
                    'message': f'商品ID {product_id} 不在分类索引中',
                    'data': None
                }), 404
        
    except Exception as e:
        logger.error(f"测试分类索引接口出现错误: {e}")
        return jsonify({
            'success': False,
            'message': f'测试失败: {str(e)}',
            'data': None
        }), 500


if __name__ == '__main__':
    # 初始化API
    if not init_api():
        logger.error("API初始化失败，退出程序")
        sys.exit(1)
    
    # 启动服务器
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # 生产环境建议关闭debug模式
    if os.getenv('ENVIRONMENT', 'development') == 'production':
        debug = False
    
    logger.info(f"启动Flask服务器 V6: {host}:{port}")
    logger.info(f"调试模式: {debug}")
    logger.info(f"环境: {os.getenv('ENVIRONMENT', 'development')}")
    
    # 启动服务器
    app.run(host=host, port=port, debug=debug, threaded=True) 