# -*- coding: utf-8 -*-
"""
File: test_data
Created on 1/8/2026 2:25 PM

Description: 
测试数据部分
@author: zzzrrr
@email: [你的邮箱]
@version: 1.0
"""
# tests/test_data.py
import pytest
import pandas as pd
import os
import shutil
from src.data.storage import SnapshotManager


# --- Fixture: 测试前的准备与测试后的清理 ---
@pytest.fixture
def temp_storage():
    """创建一个临时目录用于测试，测完自动删除"""
    path = "./tests_temp_snapshot"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    yield path  # 把路径传给测试函数

    # 测试结束后执行这里
    if os.path.exists(path):
        shutil.rmtree(path)


def test_snapshot_lifecycle(temp_storage):
    """测试：存入 -> 读取 -> 校验数据一致性"""
    mgr = SnapshotManager(base_path=temp_storage)

    # 1. 准备假数据
    df_org = pd.DataFrame({
        'price': [10.5, 11.0, 10.8],
        'vol': [100, 200, 150],
        'trade_date': ['20210101', '20210102', '20210103']
    })

    # 2. 保存
    path = mgr.save_snapshot(
        {"test_table": df_org},
        note="Pytest Run",
        extra_meta={"env": "test"}
    )

    # 3. 检查文件是否生成
    assert os.path.exists(path)
    assert os.path.exists(os.path.join(path, "manifest.json"))

    # 4. 读取
    loaded_data = mgr.load_snapshot(path)

    # 5. 比对 (Pandas 提供了专门的 assert 函数)
    pd.testing.assert_frame_equal(df_org, loaded_data['test_table'])