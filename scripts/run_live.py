# -*- coding: utf-8 -*-
"""
File: run_live
Created on 1/8/2026 1:37 PM

Description: 
[在此处添加文件描述]

@author: zzzrrr
@email: [你的邮箱]
@version: 1.0
"""
# scripts/run_live.py
import sys
import os

# 1. 计算项目根目录
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)

# 2. 【关键修改】将工作目录强制切换到项目根目录
# 这样 src 里的代码使用 "./data" 时，就会指向 Project_A/data 而不是 scripts/data
os.chdir(project_root)
print(f"工作目录已切换至: {os.getcwd()}")

# 3. 添加路径以便导入 src
sys.path.append(project_root)

from src.data.tushare_adapter import TuShareAdapter
from src.data.storage import SnapshotManager


def _format_query_source(audit: dict, default_desc: str = "未知") -> str:
    if not audit:
        return default_desc
    source = audit.get("source")
    if source == "cache":
        return "本地缓存（历史 TuShare 拉取结果）"
    if source == "tushare":
        return "TuShare 在线查询"
    if source == "error":
        err = audit.get("error", "")
        return f"TuShare 查询失败（{err}）"
    return default_desc


def live_chain_test():
    print("=== 启动在线全链路测试 (Live Integration) ===")

    # 1. 连通性测试
    print("\n[Step 1] 连接 TuShare Pro...")
    try:
        adapter = TuShareAdapter()
        print("     [OK] TuShare 适配器初始化成功")
    except Exception as e:
        print(f"     [Error] 初始化失败: {e}")
        return

    # 2. 拉取真实数据
    symbol = 'RB2105.SHF'
    start = '20210101'
    end = '20210110'
    force_online = os.getenv("LIVE_FORCE_ONLINE", "0") == "1"
    use_cache = not force_online
    print(f"\n[Step 2] 拉取真实行情 ({symbol}: {start}-{end})...")
    if force_online:
        print("     [Info] LIVE_FORCE_ONLINE=1，已关闭本地缓存，强制联网查询")
    else:
        print("     [Info] 默认模式：允许使用本地缓存（可设置 LIVE_FORCE_ONLINE=1 关闭缓存）")

    specs = adapter.get_contract_specs('SHF', use_cache=use_cache)
    specs_audit = adapter.get_last_query_audit('fut_basic')
    print(f"     合约表获取成功: {len(specs)} 条")
    print(f"     合约表来源: {_format_query_source(specs_audit)}")
    print("     合约表说明: 交易所合约基础信息更新频率较低，默认可长期复用缓存。")
    specs_cache_file = specs_audit.get("cache_file") if specs_audit else None
    if specs_cache_file:
        print(f"     合约表缓存文件: {specs_cache_file}")
        print("     如需强制刷新合约表: 删除该缓存文件后重跑，或设置 LIVE_FORCE_ONLINE=1。")

    df = adapter.get_daily_bars(symbol, start, end, use_cache=use_cache)
    daily_audit = adapter.get_last_query_audit('fut_daily')
    print(f"     日线来源: {_format_query_source(daily_audit)}")
    print("     说明: 合约表条数是该交易所可查询合约总数；行情条数是单一合约在日期区间内的交易日数量，两者不是前后截取关系。")

    if df.empty:
        print("     [Warn] 数据为空 (可能是权限问题或日期错误)")
    else:
        print(f"     [OK] 行情获取成功: {len(df)} 行")
        print(f"     数据样例: {df.iloc[0].to_dict()}")

    # 3. 真实落地验证
    print("\n[Step 3] 真实数据落地...")
    storage = SnapshotManager()
    try:
        meta = {
            "source": "TuShare Live",
            "symbol": symbol,
            "purpose": "Live Check"
        }
        path = storage.save_snapshot(
            {"live_market": df, "live_specs": specs},
            note="Live Integration Run",
            extra_meta=meta
        )
        print(f"     [OK] 快照已保存至: {path}")
        print("     (请手动检查该文件夹内的 manifest.json)")
    except Exception as e:
        print(f"     [Error] 保存失败: {e}")


if __name__ == "__main__":
    live_chain_test()
