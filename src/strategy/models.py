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
@version: 3.2
"""

import numpy as np
import pandas as pd
from .base import ModelBase
import statsmodels.tsa.stattools as ts


class RollingOLS(ModelBase):
    """
    [Baseline] 滚动 OLS 模型 (Rolling Ordinary Least Squares)

    原理:
    假设 y_t = alpha + beta * x_t + epsilon_t
    在滚动窗口内估计 beta 和 alpha，并计算残差的 Z-Score。

    特点:
    - 线性逻辑，简单直观。
    - 引入了 Rolling ADF 检验，实时监控价差的平稳性。
    """

    def __init__(self, window: int = 20, z_window: int = None,
                 use_log: bool = True):
        """
        :param window: Beta 估计窗口 (Rolling Window)
        :param z_window: Z-Score 计算窗口 (默认等于 window)
        :param use_log: 是否使用对数价格 (Log-Spread 代表百分比价差，推荐 True)
        """
        self.window = window
        self.z_window = z_window if z_window else window
        self.use_log = use_log

    def fit_predict(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        # 0. 保存原始索引 (用于最后对齐，防止 dropna 导致日期错位)
        original_index = y.index

        # 1. 预处理 (Log变换防爆盾)
        if self.use_log:
            # [Safety] 过滤 0 和 负数，防止 np.log 产生 -inf 或 NaN
            y = y.where(y > 0)
            x = x.where(x > 0)
            y = np.log(y)
            x = np.log(x)

        # 2. 数据清洗 (去除无效数据)
        df = pd.DataFrame({'y': y, 'x': x}).dropna()

        # 3. 计算滚动统计量 (Rolling Statistics)
        # Cov(x, y)
        cov = df['x'].rolling(window=self.window).cov(df['y'])
        # Var(x)
        var_x = df['x'].rolling(window=self.window).var()

        # [Robustness] 极小方差保护: 如果 x 不动(方差为0)，beta 会无穷大，强制置为 NaN
        var_x = var_x.replace(0, np.nan)

        mean_y = df['y'].rolling(window=self.window).mean()
        mean_x = df['x'].rolling(window=self.window).mean()

        # 计算参数 (OLS 公式: beta = cov/var)
        beta = cov / var_x
        alpha = mean_y - beta * mean_x

        # 4. 计算 Spread (关键步骤)
        # (A) 拟合 Spread: 使用当期 Beta (含未来信息，仅供观察拟合度)
        spread_fit = df['y'] - (alpha + beta * df['x'])

        # (B) 交易 Spread: 使用 T-1 期 Beta (严谨，无未来函数)
        # 逻辑: 今日收盘算出的 Beta，只能用于明日盘中的对冲
        beta_lag = beta.shift(1)
        alpha_lag = alpha.shift(1)
        spread_tradable = df['y'] - (alpha_lag + beta_lag * df['x'])

        # 5. Rolling ADF Test (实时协整检验)
        # 目的: 检测 Spread 是否真的具有"均值回归"特性
        adf_window = 60  # 过去 60 天
        adf_pvalues = np.full(len(df), np.nan)
        spread_values = spread_tradable.values

        for t in range(adf_window, len(df)):
            sample = spread_values[t - adf_window: t]
            sample = sample[np.isfinite(sample)]  # 去除无效值

            # [Optimization] 只有样本足够且有波动(方差>0)时才跑检验，节省算力
            if len(sample) > 20 and np.var(sample) > 1e-8:
                try:
                    # regression='c': 假设只有常数项(均值回归)，不带时间趋势
                    # maxlag=1: 简化检验，提高速度
                    res_adf = ts.adfuller(sample, maxlag=1, regression='c',
                                          autolag=None)
                    adf_pvalues[t] = res_adf[1]  # 取 p-value
                except:
                    pass  # 计算失败则保持 NaN

        # 6. 计算 Z-Score (标准化)
        spread_rolling = spread_tradable.rolling(window=self.z_window)
        spread_mean = spread_rolling.mean()
        # [Robustness] 同样防止 std=0 导致 inf
        spread_std = spread_rolling.std().replace(0, np.nan)

        z_score = (spread_tradable - spread_mean) / spread_std

        # 7. 构造结果表
        res = pd.DataFrame({
            'beta': beta,  # 当期估计 (后验)
            'beta_lag': beta_lag,  # 交易使用 (先验)
            'spread': spread_tradable,  # 可交易价差
            'z_score': z_score,  # 交易信号源
            'spread_std': spread_std,  # 波动率 (用于风控)
            'adf_p': adf_pvalues  # 协整度 (用于闸门)
        }, index=df.index)

        # [Safety] 标记有效性:
        # 必须同时满足: Z-Score 是有限数(非inf/nan) 且 Beta_lag 存在
        res['valid'] = np.isfinite(res['z_score']) & res['beta_lag'].notna()

        # 8. 对齐与类型强制 (最后的防线)
        res = res.reindex(original_index)
        # [Fix] 强制填充 False 并转为 bool 类型，防止下游 if 判断报错
        res['valid'] = res['valid'].fillna(False).astype(bool)

        return res


class SimpleKalman(ModelBase):
    """
    [Enhanced] 自适应卡尔曼滤波 (Adaptive Kalman Filter)

    原理:
    将 Beta 视为一个随时间随机游走的状态变量。
    State Equation: beta_t = beta_{t-1} + noise (Q)
    Obs Equation:   y_t = beta_t * x_t + noise (R)

    特点:
    - 自适应: R 会根据最近的预测误差自动调整 (Adaptive)。
    - 快速反应: 比 OLS 更快捕捉到 Beta 的结构性变化。
    """

    def __init__(self, delta=1e-5, ve=1e-3, z_window=30, adaptive=False,
                 use_log: bool = True):
        """
        :param delta: 状态转移协方差 (Q)，控制 Beta 变化的灵活性 (越大越敏感)
        :param ve: 观测噪声方差 (R) 的初始值
        :param adaptive: 是否开启自适应 R (推荐 True)
        """
        self.delta = delta
        self.ve = ve
        self.z_window = z_window
        self.adaptive = adaptive
        self.use_log = use_log

    def fit_predict(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        original_index = y.index

        # 1. 预处理 (Log变换)
        if self.use_log:
            y = y.where(y > 0)
            x = x.where(x > 0)
            y = np.log(y)
            x = np.log(x)

        df_input = pd.DataFrame({'y': y, 'x': x}).dropna()
        Y = df_input['y'].values
        X = df_input['x'].values
        n = len(Y)

        # 初始化状态 [alpha, beta]
        # 状态向量 state = [[alpha], [beta]]
        state = np.zeros((2, 1))

        # P: 状态协方差矩阵。初始化为大数(1000)，代表对初始值(0,0)完全不信任，
        # 迫使模型快速根据前几个观测值收敛。
        P = np.eye(2) * 1000.0

        beta_arr = np.full(n, np.nan)
        alpha_arr = np.full(n, np.nan)
        spread_arr = np.full(n, np.nan)

        Q = np.eye(2) * self.delta  # 过程噪声矩阵
        R = self.ve  # 观测噪声 (标量)

        error_window = []
        adaptive_lookback = 30  # 自适应回看窗口

        # --- Kalman Filter Loop ---
        for t in range(n):
            H = np.array([[1.0, X[t]]])  # 观测矩阵

            # A. 预测步骤 (Time Update)
            # 基于 T-1 的状态预测 T
            state_pred = state.copy()
            P_pred = P + Q

            # 计算预测误差 (Innovation / Pre-fit Residual)
            # 这里的 y_hat 完全基于过去信息，因此 spread 是 Tradable 的
            y_hat = float(np.dot(H, state_pred))
            e = Y[t] - y_hat

            # B. 自适应逻辑 (Adaptive R)
            if self.adaptive and len(error_window) > 10:
                # 如果最近预测误差变大，说明市场波动加剧，增大 R
                current_vol = np.var(error_window[-adaptive_lookback:])
                R = max(current_vol, 1e-5)
            error_window.append(e)

            # C. 更新步骤 (Measurement Update)
            # S: Innovation Covariance
            S = float(np.dot(H, np.dot(P_pred, H.T))) + R
            # [Stability] 熔断保护: 防止 S 接近 0 导致除零错误
            S = max(S, 1e-12)

            # K: Kalman Gain
            K = np.dot(P_pred, H.T) / S

            # 更新状态 (Posterior)
            state = state_pred + K * e
            P = (np.eye(2) - np.dot(K, H)) @ P_pred

            # 记录结果
            alpha_arr[t] = state[0, 0]
            beta_arr[t] = state[1, 0]
            spread_arr[t] = e  # 记录 Innovation 作为 Spread

        # 构造基础结果
        res = pd.DataFrame({
            'beta': beta_arr,
            'alpha': alpha_arr,
            'spread': spread_arr
        }, index=df_input.index)

        # [Safety] 构造 beta_lag
        # 虽然 Kalman Spread 是 tradable 的，但为了和 OLS 接口统一，
        # 且仓位计算通常需要确定的 Hedge Ratio，我们强制输出滞后一期的 Beta
        res['beta_lag'] = res['beta'].shift(1)

        # Rolling ADF Check (同 OLS)
        adf_window = 60
        adf_pvalues = np.full(len(res), np.nan)
        spread_vals = res['spread'].values
        for t in range(adf_window, len(res)):
            sample = spread_vals[t - adf_window: t]
            sample = sample[np.isfinite(sample)]
            if len(sample) > 20 and np.var(sample) > 1e-8:
                try:
                    res_adf = ts.adfuller(sample, maxlag=1, regression='c',
                                          autolag=None)
                    adf_pvalues[t] = res_adf[1]
                except:
                    pass
        res['adf_p'] = adf_pvalues

        # 计算 Z-Score
        spread_rolling = res['spread'].rolling(window=self.z_window)
        res['spread_std'] = spread_rolling.std().replace(0, np.nan)
        res['z_score'] = (res['spread'] - spread_rolling.mean()) / res[
            'spread_std']

        # [Safety] 标记有效性
        res['valid'] = np.isfinite(res['z_score']) & res['beta_lag'].notna()

        # [Warmup] 强制前10个点无效 (卡尔曼初期收敛不稳定)
        if len(res) > 10:
            valid_col_idx = res.columns.get_loc('valid')
            res.iloc[:10, valid_col_idx] = False

        # 对齐与类型强制
        res = res.reindex(original_index)
        res['valid'] = res['valid'].fillna(False).astype(bool)

        return res