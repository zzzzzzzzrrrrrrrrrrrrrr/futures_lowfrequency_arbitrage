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
    x = pd.Series(np.arange(100, dtype=float))
    y = x * 2.0

    # 2. 运行模型 (window=10)
    model = RollingOLS(window=10)
    res = model.fit_predict(y, x)

    # 3. 断言 (Assert)
    # 取最后一个点的 beta，应该非常接近 2.0
    last_beta = res['beta'].iloc[-1]
    last_z = res['z_score'].iloc[-1]

    assert last_beta == pytest.approx(2.0,
                                      abs=0.001), f"Beta算错了: {last_beta}"
    # 完美线性关系，残差为0，Z-score 理论上是 NaN 或 0 (取决于浮点误差)，这里主要测代码不崩
    assert 'valid' in res.columns


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