# -*- coding: utf-8 -*-
"""
File: run_smoke
Created on 1/8/2026 1:37 PM

Description: 
[在此处添加文件描述]

@author: zzzrrr
@email: [你的邮箱]
@version: 1.0
"""
# scripts/run_smoke.py
import sys
import os
import pandas as pd
import numpy as np
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.strategy.models import RollingOLS, SimpleKalman
from src.data.storage import SnapshotManager


def smoke_test_models():
    print("\n[Smoke] 1. 测试数学模型逻辑 (使用合成数据)...")

    length = 100

    # --- 修正造数逻辑 ---
    # 1. 先造 X (Leg2)，让它“干净”一点，没有额外噪音
    leg2 = pd.Series(np.random.randn(length) + 100, name='leg2')

    # 2. 再造 Y (Leg1)，把噪音加在 Y 上
    # Y = 2.0 * X + Noise
    # 这样满足 OLS 的 y = beta*x + epsilon 假设，估计量是无偏的
    leg1 = leg2 * 2.0 + np.random.normal(0, 0.1, length)
    leg1.name = 'leg1'
    # --------------------

    # 2. 测试 OLS
    print("   -> 运行 RollingOLS...")
    ols = RollingOLS(window=20)
    # 注意: y=leg1, x=leg2
    res_ols = ols.fit_predict(leg1, leg2)
    beta = res_ols['beta'].iloc[-1]

    # 验证 Beta 是否接近 2.0
    print(f"      OLS Beta: {beta:.4f} (预期接近 2.0)")
    if 1.95 < beta < 2.05:  # 此时应该非常准，阈值可以收紧
        print("      [PASS] OLS Beta 正常")
    else:
        print("      [FAIL] OLS Beta 异常")

    # 3. 测试 Kalman
    print("   -> 运行 SimpleKalman (adaptive=True)...")
    kf = SimpleKalman(adaptive=True)
    res_kf = kf.fit_predict(leg1, leg2)

    if not res_kf.empty and res_kf['beta'].notna().any():
        final_beta_kf = res_kf['beta'].iloc[-1]
        print(
            f"      [PASS] Kalman 运行正常 (Final Beta: {final_beta_kf:.4f})")
    else:
        print("      [FAIL] Kalman 输出无效")


def smoke_test_storage():
    print("\n[Smoke] 2. 测试存储模块 (本地临时读写)...")
    test_path = os.path.join(project_root, "data_snapshot_smoke")
    mgr = SnapshotManager(base_path=test_path)

    df = pd.DataFrame({'price': [10, 11, 12], 'vol': [100, 200, 300]})

    try:
        path = mgr.save_snapshot({"smoke_test": df}, note="Smoke Run")
        loaded = mgr.load_snapshot(path)
        if len(loaded['smoke_test']) == 3:
            print(f"      [PASS] 存取校验通过，路径: {path}")
        else:
            print("      [FAIL] 数据行数不匹配")

        shutil.rmtree(path)
        print("      (已自动清理测试文件)")

    except Exception as e:
        print(f"      [FAIL] 存储测试失败: {e}")


if __name__ == "__main__":
    print("=== 启动离线冒烟测试 (Offline Smoke Test) ===")
    smoke_test_models()
    smoke_test_storage()
    print("\n=== 测试结束 ===")