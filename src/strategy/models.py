# -*- coding: utf-8 -*-
"""
File: models
Created on 12/9/2025 1:01 PM

Description: 
核心数学模型库
负责计算对冲比率(Beta)、Alpha、Spread 以及 Z-Score。
包含:
1. RollingOLS: 滚动窗口最小二乘 (Baseline)
2. AdaptiveKalman: 自适应卡尔曼滤波 (Enhanced)

Updates v2.2:
1. [RollingOLS] 引入 spread_tradable (基于 lag beta)，消除 Look-ahead Bias。
2. [RollingOLS] 增加方差除零保护。
3. [Kalman] 增加数据预处理 (对齐 + dropna)。
4. [Kalman] 优化 P 矩阵初始化 (加速收敛)。
5. [Kalman] 修复 warmup 阶段 iloc 切片 bug。

Updates v2.3:
1. [Safety] 修复 Z-Score 可能为 inf 的严重 bug (使用 np.isfinite 替代 notna)。
2. [Safety] RollingOLS valid 判定强制依赖 beta_lag (确保无未来函数)。
3. [Safety] Kalman valid 判定增加 finite 检查。
@author: zzzrrr
@email: [你的邮箱]
@version: 2.3
"""

import numpy as np
import pandas as pd


class PairModelBase:
    """配对交易模型基类"""

    def fit_predict(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        raise NotImplementedError


class RollingOLS(PairModelBase):
    """
    [Baseline] 滚动 OLS 模型
    特点: 简单、参数少。
    注意: Z-Score 计算基于 spread_tradable (无未来函数)。
    """

    def __init__(self, window: int = 20, z_window: int = None):
        """
        :param window: Beta 估计窗口
        :param z_window: Z-Score 计算窗口 (默认等于 window)
        """
        self.window = window
        self.z_window = z_window if z_window else window

    def fit_predict(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        # 1. 数据对齐与清洗
        df = pd.DataFrame({'y': y, 'x': x}).dropna()

        # 2. 计算滚动统计量 (Rolling Window)
        # Cov(x, y) & Var(x)
        # 使用 Series.rolling 防止 index 对齐问题
        cov = df['x'].rolling(window=self.window).cov(df['y'])
        var_x = df['x'].rolling(window=self.window).var()

        # [Robustness] 极小方差保护，防止除零爆炸
        var_x = var_x.replace(0, np.nan)

        mean_y = df['y'].rolling(window=self.window).mean()
        mean_x = df['x'].rolling(window=self.window).mean()

        # 3. 计算参数 (当期拟合值)
        beta = cov / var_x
        alpha = mean_y - beta * mean_x

        # 4. 计算 Spread
        # (A) 拟合 Spread: 当期 Beta (含未来信息，仅供观察)
        spread_fit = df['y'] - (alpha + beta * df['x'])

        # (B) 交易 Spread: T-1 期 Beta (严谨，用于实盘)
        beta_lag = beta.shift(1)
        alpha_lag = alpha.shift(1)
        spread_tradable = df['y'] - (alpha_lag + beta_lag * df['x'])

        # 5. 计算 Z-Score (基于可交易 Spread)
        spread_rolling = spread_tradable.rolling(window=self.z_window)
        spread_mean = spread_rolling.mean()
        # [Robustness] std 可能为 0 (极小概率但存在)，replace 0 -> nan 避免 inf
        spread_std = spread_rolling.std().replace(0, np.nan)

        z_score = (spread_tradable - spread_mean) / spread_std

        # 6. 构造结果
        res = pd.DataFrame({
            'beta': beta,  # 当期估计
            'beta_lag': beta_lag,  # 实际使用
            'spread_fit': spread_fit,  # 观察用
            'spread': spread_tradable,  # 交易用
            'z_score': z_score,
            'spread_std': spread_std
        }, index=df.index)

        # [Safety] 标记有效性: 必须是有限数 (Finite) 且 lag 参数存在
        # isfinite 同时排除了 NaN 和 Inf
        res['valid'] = np.isfinite(res['z_score']) & res['beta_lag'].notna()

        return res


class SimpleKalman(PairModelBase):
    """
    [Enhanced] 自适应卡尔曼滤波 (Adaptive Kalman Filter)
    """

    def __init__(self, delta=1e-5, ve=1e-3, z_window=30, adaptive=False):
        self.delta = delta
        self.ve = ve
        self.z_window = z_window
        self.adaptive = adaptive

    def fit_predict(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        # 1. [Robustness] 强制数据对齐与清洗
        df_input = pd.DataFrame({'y': y, 'x': x}).dropna()
        Y = df_input['y'].values
        X = df_input['x'].values
        n = len(Y)

        # 初始化状态 [alpha, beta]
        state = np.zeros((2, 1))

        # [Optimization] P 初始化
        P = np.eye(2) * 1000.0

        beta_arr = np.full(n, np.nan)
        alpha_arr = np.full(n, np.nan)
        spread_arr = np.full(n, np.nan)

        Q = np.eye(2) * self.delta
        R = self.ve

        error_window = []
        adaptive_lookback = 30

        for t in range(n):
            H = np.array([[1.0, X[t]]])

            # Predict
            state_pred = state.copy()
            P_pred = P + Q

            # Innovation (Spread) - 天然 Tradable (基于 T-1)
            y_hat = float(np.dot(H, state_pred))
            e = Y[t] - y_hat

            # Adaptive R
            if self.adaptive and len(error_window) > 10:
                current_vol = np.var(error_window[-adaptive_lookback:])
                R = max(current_vol, 1e-5)
            error_window.append(e)

            # Update
            S = float(np.dot(H, np.dot(P_pred, H.T))) + R
            K = np.dot(P_pred, H.T) / S
            state = state_pred + K * e
            P = (np.eye(2) - np.dot(K, H)) @ P_pred

            alpha_arr[t] = state[0, 0]
            beta_arr[t] = state[1, 0]
            spread_arr[t] = e

        res = pd.DataFrame({
            'beta': beta_arr,
            'alpha': alpha_arr,
            'spread': spread_arr
        }, index=df_input.index)

        # Z-Score
        spread_rolling = res['spread'].rolling(window=self.z_window)
        # [Safety] 防止 std=0 导致 inf
        res['spread_std'] = spread_rolling.std().replace(0, np.nan)
        res['z_score'] = (res['spread'] - spread_rolling.mean()) / res[
            'spread_std']

        # [Safety] 标记有效性: 必须是有限数
        res['valid'] = np.isfinite(res['z_score']) & res['beta'].notna()

        # [Warmup] 强制前10个点无效 (使用 iloc 安全切片)
        if len(res) > 10:
            valid_col_idx = res.columns.get_loc('valid')
            res.iloc[:10, valid_col_idx] = False

        return res