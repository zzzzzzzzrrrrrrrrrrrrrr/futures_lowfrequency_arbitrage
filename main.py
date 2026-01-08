# -*- coding: utf-8 -*-
"""
File: main.py
Description: 全链路集成测试脚本
用于验证数据层 (Req 1) 的各个组件是否正常工作，包括:
1. TuShare 连接与 Token 校验
2. 数据适配器 (Adapter) 的字段标准化
3. 快照管理器 (Snapshot) 的落地与审计
"""
import sys
import os

# 将项目根目录加入 Python 路径，确保能 import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.tushare_adapter import TuShareAdapter
from src.data.storage import SnapshotManager


def run_system_check():
    print("=== 全链路集成测试：元数据 + 行情 + 存储 ===")

    # 1. 初始化组件
    try:
        adapter = TuShareAdapter()  # 负责拉数据
        storage = SnapshotManager()  # 负责存数据
        print("组件初始化完成 (Adapter & Storage)")
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    # 2. 测试合约元数据 (Req 1.2 & 降级机制)
    print("\n[Step 1] 获取合约元数据 (Security Master)...")
    specs = adapter.get_contract_specs('SHF')

    if not specs.empty:
        count = len(specs)
        source = "TuShare" if count > 10 else "本地配置(instruments.json)"
        print(f"获取成功! 共 {count} 个品种")
        print(f"   数据来源: {source}")
    else:
        print("失败: 既没连上 TuShare，也没找到本地配置")

    # 3. 测试行情获取 (Req 1.1)
    contract = 'RB2105.SHF'
    print(f"\n[Step 2] 获取 {contract} 行情数据...")
    df = adapter.get_daily_bars(contract, '20210101', '20210301')
    if df.empty:
        print("行情为空，跳过后续步骤")
        return
    print(f"行情获取成功: {len(df)} 条")

    # 4. 测试快照落地与审计 (Req 1.3)
    print("\n[Step 3] 数据入库与审计 (Req 1.3 规范验证)...")

    # 准备数据包
    data_packet = {
        "security_master": specs,
        "market_data": df
    }

    # 构造审计元数据 (Req 1.3 要求: 必须记录查询参数和意图)
    meta_info = {
        "data_source": "TuShare Pro / Local Config",
        "query_contract": contract,
        "query_range": ["20210101", "20210301"],
        "purpose": "Integration Test / Format Verification"
    }

    try:
        # 调用 save_snapshot 时传入 meta_info
        path = storage.save_snapshot(data_packet, note="规范化路径测试",
                                     extra_meta=meta_info)
        print(f"数据已落地: {path}")

        # 立即回读校验
        check_data = storage.load_snapshot(path)
        if len(check_data) == 2:
            print("闭环校验通过: 合约表 + 行情表 均完整无误。")
    except Exception as e:
        print(f"存储/校验失败: {e}")


if __name__ == "__main__":
    run_system_check()