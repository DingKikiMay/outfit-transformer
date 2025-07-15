#!/usr/bin/env python3
"""
清理旧的checkpoint文件，释放系统盘空间
"""

import os
import shutil
import glob

def cleanup_checkpoints():
    """清理checkpoint文件"""
    
    # 清理outfit-transformer目录下的checkpoints
    checkpoint_dir = "/root/outfit-transformer/checkpoints"
    if os.path.exists(checkpoint_dir):
        print(f"清理目录: {checkpoint_dir}")
        try:
            shutil.rmtree(checkpoint_dir)
            print(f"✓ 已删除 {checkpoint_dir}")
        except Exception as e:
            print(f"✗ 删除失败: {e}")
    
    # 清理其他可能的checkpoint目录
    other_checkpoint_dirs = [
        "/root/autodl-tmp/checkpoints",
        "/root/outfit-transformer/src/run/checkpoints"
    ]
    
    for dir_path in other_checkpoint_dirs:
        if os.path.exists(dir_path):
            print(f"清理目录: {dir_path}")
            try:
                shutil.rmtree(dir_path)
                print(f"✓ 已删除 {dir_path}")
            except Exception as e:
                print(f"✗ 删除失败: {e}")
    
    # 清理Python缓存文件
    cache_dirs = [
        "/root/outfit-transformer/__pycache__",
        "/root/outfit-transformer/src/__pycache__",
        "/root/outfit-transformer/src/run/__pycache__"
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            print(f"清理缓存: {cache_dir}")
            try:
                shutil.rmtree(cache_dir)
                print(f"✓ 已删除 {cache_dir}")
            except Exception as e:
                print(f"✗ 删除失败: {e}")
    
    # 清理conda缓存
    conda_cache = "/root/miniconda3/pkgs"
    if os.path.exists(conda_cache):
        print(f"清理conda缓存: {conda_cache}")
        try:
            # 只删除下载的包文件，保留已安装的包
            for item in os.listdir(conda_cache):
                item_path = os.path.join(conda_cache, item)
                if os.path.isfile(item_path) and item.endswith('.tar.bz2'):
                    os.remove(item_path)
                    print(f"✓ 已删除 {item}")
        except Exception as e:
            print(f"✗ 清理conda缓存失败: {e}")
    
    print("\n清理完成！现在可以重新开始训练了。")
    print("新的训练策略：")
    print("1. 不保存中间checkpoint，只在训练结束时保存最佳模型")
    print("2. 最佳模型保存到数据盘 /root/autodl-tmp/")
    print("3. 大幅减少磁盘占用，避免系统盘满")

if __name__ == "__main__":
    cleanup_checkpoints() 