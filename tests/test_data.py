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
import json
import shutil
from src.data.storage import SnapshotManager
from src.data.tushare_adapter import TuShareAdapter


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


def test_snapshot_csv_hash_compatibility(temp_storage, monkeypatch):
    """CSV fallback should use file hash and remain loadable."""
    mgr = SnapshotManager(base_path=temp_storage)
    df_org = pd.DataFrame({
        'price': [10.5, 11.0, 10.8],
        'vol': [100, 200, 150],
        'trade_date': ['20210101', '20210102', '20210103']
    })

    def _force_csv(*args, **kwargs):
        raise RuntimeError("force csv fallback")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _force_csv)
    path = mgr.save_snapshot({"test_table": df_org}, note="csv hash test")

    with open(os.path.join(path, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["files"][0]["file"].endswith(".csv")
    assert manifest["files"][0]["hash_type"] == "file_sha256"

    loaded = mgr.load_snapshot(path)
    pd.testing.assert_frame_equal(df_org, loaded["test_table"], check_dtype=False)


def test_get_daily_bars_requires_contract_columns():
    """Missing required contract columns should raise instead of silent degrade."""
    adapter = TuShareAdapter.__new__(TuShareAdapter)
    adapter._safe_query = lambda *args, **kwargs: pd.DataFrame({
        "trade_date": ["20260105"],
        "open": [10.0],
        "high": [11.0],
        "low": [9.5],
        "close": [10.5],
        "vol": [1000],
        "oi": [2000]
    })

    with pytest.raises(ValueError, match="缺少必需字段"):
        adapter.get_daily_bars("RB2605.SHF", "20260101", "20260110")


def test_get_chain_resolves_symbol_to_exchange():
    """get_chain should resolve product symbol -> exchange and then filter chain."""
    adapter = TuShareAdapter.__new__(TuShareAdapter)
    adapter._load_local_config = lambda: {
        "contracts": {
            "RB": {"exchange": "SHF"}
        }
    }

    called = {}

    def _fake_get_contract_specs(query_key):
        called["query_key"] = query_key
        return pd.DataFrame([
            {"symbol": "RB", "ts_code": "RB2405.SHF", "expiry_date": "20240515"},
            {"symbol": "RB", "ts_code": "RB2410.SHF", "expiry_date": "20241015"},
            {"symbol": "HC", "ts_code": "HC2405.SHF", "expiry_date": "20240515"}
        ])

    adapter.get_contract_specs = _fake_get_contract_specs
    adapter._safe_query = lambda *args, **kwargs: pd.DataFrame(
        [{"mapping_ts_code": "RB2405.SHF"}]
    )

    chain = adapter.get_chain("RB", "20240101")

    assert called["query_key"] == "SHF"
    assert chain["all"] == ["RB2405.SHF", "RB2410.SHF"]
    assert chain["main"] == "RB2405.SHF"
    assert chain["submain"] == "RB2410.SHF"
    assert chain["near"] == "RB2405.SHF"
    assert chain["next"] == "RB2410.SHF"
