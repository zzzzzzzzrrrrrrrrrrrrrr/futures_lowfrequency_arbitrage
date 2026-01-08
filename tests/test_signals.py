# -*- coding: utf-8 -*-
"""
File: test_signals
Created on 1/8/2026 2:26 PM

Description: 
[在此处添加文件描述]

@author: zzzrrr
@email: [你的邮箱]
@version: 1.0
"""
# tests/test_signals.py
import pytest
import pandas as pd
import numpy as np
from src.strategy.signals import SignalGenerator


def test_signal_entry_exit():
    """
    测试信号生成器的核心逻辑：
    1. Z-Score 突破阈值 -> 开仓
    2. Z-Score 回归 -> 平仓
    """
    # 1. 构造场景:
    # Leg1 先平稳，然后暴涨(触发Short)，然后回归(触发Exit)
    # 使用 50 个点
    leg2 = pd.Series([100.0] * 50)  # Leg2 不动
    leg1 = pd.Series([100.0] * 50)  # Leg1 初始也不动

    # 第 20-30 天暴涨 (Spread变大 -> Z变大 -> 应该做空Pair: Short Leg1)
    leg1.iloc[20:30] = 110.0

    # 第 30 天后回归
    leg1.iloc[30:] = 100.0

    # 2. 运行生成器
    # 阈值: 开仓=2.0, 平仓=0.5
    # 注意: OLS window=5 保证反应快
    gen = SignalGenerator(model_type='ols', window=5,
                          entry_threshold=2.0, exit_threshold=0.5,
                          cooling_days=0)

    res = gen.run(leg1, leg2)

    # 3. 验证逻辑
    signals = res['signal']

    # A. 检查开仓: 在暴涨区间(20-30)，应该出现 -1 (做空价差)
    # 因为有 window=5 的滞后，信号可能在 25 左右才出来
    has_short = (signals.iloc[20:35] == -1).any()
    assert has_short, "Z-Score 暴涨时未触发做空信号"

    # B. 检查平仓: 在回归区间(35以后)，应该变回 0
    # 同样考虑到滞后，看最后几行
    last_signal = signals.iloc[-1]
    assert last_signal == 0, "价差回归后未平仓"