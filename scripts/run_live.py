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
    print(f"\n[Step 2] 拉取真实行情 ({symbol}: {start}-{end})...")

    specs = adapter.get_contract_specs('SHF')
    print(f"     合约表获取成功: {len(specs)} 条")

    df = adapter.get_daily_bars(symbol, start, end)

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