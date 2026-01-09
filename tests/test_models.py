# -*- coding: utf-8 -*-
"""
File: test_models
Created on 1/8/2026 2:24 PM

Description: 测试模型部分


@author: zzzrrr
@email: [你的邮箱]
@version: 1.0
"""
# tests/test_models.py
import pytest
import numpy as np
import pandas as pd
from src.strategy.models import RollingOLS, SimpleKalman


def test_rolling_ols_perfect_linear():
    """测试 OLS 能否识别完美的 2倍 线性关系"""
    # 1. 造数据: y = 2 * x
    x = pd.Series(np.arange(100, dtype=float) + 10)  # 加10防止 log(0)
    y = x * 2.0

    # 2. 运行模型 (显式关闭 Log，测试纯线性回归逻辑)
    model = RollingOLS(window=10, use_log=False)  # <--- 关键修改
    res = model.fit_predict(y, x)

    # 3. 断言
    last_beta = res['beta'].iloc[-1]
    # 在线性模式下，Beta 应该等于 2.0
    assert last_beta == pytest.approx(2.0,
                                      abs=0.001), f"Beta算错了: {last_beta}"


def test_kalman_structure():
    """测试 Kalman 输出格式是否完整"""
    x = pd.Series(np.random.randn(50) + 10)
    y = pd.Series(np.random.randn(50) + 10)

    # 测试自适应模式
    kf = SimpleKalman(adaptive=True)
    res = kf.fit_predict(y, x)

    # 检查列名
    expected_cols = ['beta', 'alpha', 'spread', 'z_score', 'spread_std',
                     'valid']
    for col in expected_cols:
        assert col in res.columns, f"Kalman 缺少列: {col}"

    # 检查 Warmup (前10个应该无效)
    assert res['valid'].iloc[0] == False
    # 后面的应该有效 (除非全是 NaN)
    assert res['valid'].iloc[-1] == True