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
import pandas as pd
import numpy as np
from src.strategy.rules import StrategyConfig
from src.strategy.models import RollingOLS
from src.strategy.signals import PairSignal


def test_signal_entry_exit_integration_honest():
    """
    目标: 构造一段策略定义的行情，在不修改任何中间变量的情况下，
    让 RollingOLS 算出的指标自然通过 Signals 的所有风控闸门。

    1. 不能触发 Beta Jump (价差必须渐变)
    2. 不能触发 ADF (序列必须有均值回归特性)
    3. 必须触发 Z-Score (幅度要够大)
    """
    np.random.seed(42)  # 固定随机种子
    n = 300  # 需要足够长的数据让 ADF 收敛

    # 1. 构造基准 (Leg2): 随机游走
    leg2 = pd.Series(np.cumsum(np.random.normal(0, 1, n)) + 100)

    # 2. 构造完美的 Spread (均值回归信号)
    # 我们用一个正弦波来模拟 Spread，因为它天然是均值回归的，ADF 绝对能过
    # 周期设为 100 天，幅度设为 5.0 (很大)
    t = np.linspace(0, 4 * np.pi, n)  # 2个完整周期
    spread_signal = 5.0 * np.sin(t)

    # 3. 合成 Leg1
    # Leg1 = 1.0 * Leg2 + Spread
    # 这样 Beta 理论上恒定为 1.0，不会触发 Stability Gate
    leg1 = leg2 + spread_signal

    # 4. 配置 (使用严格的默认参数)
    cfg = StrategyConfig(
        model_type='ols',
        window=20,  # 窗口
        entry_threshold=1.5,  # 稍微降低一点门槛，方便测试
        exit_threshold=0.0,
        use_log=False,  # 简单线性
        beta_jump_limit=0.5,  # 严格风控
        adf_threshold=0.05,  # 严格协整要求
        vol_target=None
    )

    # 5. 运行模型 (Math Layer)
    # 这里我们完全不干预结果
    model = RollingOLS(window=cfg.window, use_log=cfg.use_log)
    df_indicators = model.fit_predict(leg1, leg2)

    # 6. 运行信号 (Logic Layer)
    signal_gen = PairSignal(cfg)
    res = signal_gen.run(df_indicators)

    # 7. 验证
    signals = res['signal']

    # 调试: 看看算出来的 ADF 是多少 (应该很小)
    # print(df_indicators[['spread', 'z_score', 'adf_p', 'beta_lag']].iloc[50:100])

    # A. 验证是否开仓
    # 正弦波在波峰处 (spread > 0) 应该做空 (-1)
    # 在波谷处 (spread < 0) 应该做多 (1)

    # 找到 spread 最宽的区域 (波峰)
    # 你的模型有滞后，所以信号会晚一点
    has_short = (signals == -1).any()
    has_long = (signals == 1).any()

    assert has_short, "构建了完美正弦波 Spread，却没触发做空信号！(可能是 Beta 风控误杀)"
    assert has_long, "构建了完美正弦波 Spread，却没触发做多信号！"

    # B. 验证平仓
    # 在正弦波穿越 0 轴附近，信号应归 0
    # 简单的检查方法：看信号是否发生过翻转 (-1 -> 0 -> 1)
    transitions = signals.diff().abs().sum()
    assert transitions > 2, "信号没有进行开平仓轮转"


def test_signal_logic_only():
    """
    [纯逻辑测试]
    只测试 'Signals类' 的 if/else 逻辑，
    不测试 'Models类' 的计算能力。
    """
    z_vals = np.concatenate([
        np.zeros(10),
        np.full(5, 3.0),
        np.full(5, 1.0),
        np.zeros(5)
    ])

    df_mock = pd.DataFrame({
        'z_score': z_vals,
        'valid': True,
        'spread_std': 0.1,
        'beta_lag': 1.0,
        'adf_p': 0.001  # 极显著
    })

    cfg = StrategyConfig(entry_threshold=2.0, exit_threshold=0.5)

    signal_gen = PairSignal(cfg)
    res = signal_gen.run(df_mock)

    s = res['signal'].values

    assert s[11] == -1
    assert s[16] == -1
    assert s[21] == 0