# -*- coding: utf-8 -*-
"""
File: src/strategy/signals.py
Description:
策略信号核心逻辑 (v5.0 - Live Ready)
Updates:
- [Fix] 稳定性闸门改为使用 beta_lag (可交易性修复)。
- [Fix] Invalid 状态改为 Hold (维持仓位)，避免数据抖动导致磨损。
- [Feature] 输出 vol_scalar (波动率倒数加权) 实现仓位管理。
"""
import pandas as pd
import numpy as np
from .base import SignalBase
from .rules import StrategyConfig


class PairSignal(SignalBase):
    def __init__(self, config: StrategyConfig):
        self.cfg = config

    def run(self, indicators: pd.DataFrame) -> pd.DataFrame:
        """
        :param indicators: 含 [z_score, valid, spread_std, adf_p, beta_lag]
        """
        df = indicators.copy()
        n = len(df)

        # 1. 结果容器
        signals = np.zeros(n)
        states = np.zeros(n)
        # [新增] 仓位缩放系数 (默认为 1.0)
        vol_scalars = np.ones(n)

        # 2. 提取数据 (Vectorization)
        if 'z_score' not in df.columns:
            raise ValueError("Missing 'z_score'")

        z_scores = df['z_score'].values
        valid_flags = df['valid'].values if 'valid' in df.columns else np.ones(
            n, bool)
        vols = df['spread_std'].values if 'spread_std' in df.columns else None
        adf_p = df['adf_p'].values if 'adf_p' in df.columns else np.zeros(n)

        # [Fix] 稳定性检查: 使用 beta_lag (T-1) 而不是 beta (T)
        # 这样在 T 时刻收盘时，我们就能确定"明天能不能开仓"
        beta_lag_series = df[
            'beta_lag'] if 'beta_lag' in df.columns else pd.Series(0,
                                                                   index=df.index)
        # 计算 lagged beta 的跳变
        beta_jump_series = beta_lag_series.diff().abs().fillna(0)
        beta_jumps = beta_jump_series.values

        current_pos = 0
        cooldown_counter = 0
        holding_timer = 0

        for t in range(n):
            # --- A. 计算仓位缩放 (Position Sizing) ---
            # 逻辑: 波动越大，仓位越小 (Risk Parity 思想)
            if self.cfg.vol_target and vols is not None:
                curr_vol = vols[t]
                # 防止除零和极端值
                if np.isfinite(curr_vol) and curr_vol > 1e-8:
                    # Scalar = Target / Actual
                    # 例如: 目标波动 1%, 实际 2% -> 仓位 0.5
                    scaler = self.cfg.vol_target / curr_vol
                    # [Safety] 限制杠杆上限，例如最大放大 3 倍，最小 0.1
                    scaler = np.clip(scaler, 0.1, 3.0)
                    vol_scalars[t] = scaler
                else:
                    vol_scalars[t] = 0.0  # 波动率异常，不给仓位

            # --- B. 有效性检查 (Soft Invalid Logic) ---
            # 如果当前 Bar 数据无效 (比如 NaN, Inf, 或 Warmup 期)
            if not valid_flags[t] or not np.isfinite(z_scores[t]):
                # [Fix] 不再强制平仓，而是维持上一时刻状态 (Hold)
                # 这避免了数据偶发丢失导致的无谓平仓
                signals[t] = current_pos
                # 状态标记为 10 (Invalid Hold) 用于审计
                states[t] = 10
                # 计时器暂停或继续取决于你的偏好，这里选择暂停
                continue

            # --- C. 冷却期 ---
            if cooldown_counter > 0:
                cooldown_counter -= 1
                signals[t] = 0
                states[t] = 9
                current_pos = 0
                holding_timer = 0
                continue

            # --- D. 计时器 ---
            if current_pos != 0:
                holding_timer += 1
            else:
                holding_timer = 0

            # --- E. 风控 (Gates) ---

            # [1] 结构止损
            if abs(z_scores[t]) > self.cfg.stop_loss:
                if current_pos != 0:
                    current_pos = 0
                    cooldown_counter = self.cfg.cooling_days * 2
                    states[t] = 8
                signals[t] = 0
                continue

            # [2] 时间止损
            if current_pos != 0 and holding_timer > self.cfg.max_holding_days:
                current_pos = 0
                cooldown_counter = self.cfg.cooling_days
                states[t] = 6
                signals[t] = 0
                continue

            # [3] 波动率过滤 (Regime)
            if self.cfg.vol_filter and vols is not None:
                if vols[t] > self.cfg.vol_filter:
                    current_pos = 0
                    signals[t] = 0
                    cooldown_counter = 1
                    states[t] = 7
                    continue

            # [4] 稳定性闸门 (Lagged Beta Jump)
            # 使用 lag 数据，确保是"事前"风控
            is_unstable = beta_jumps[t] > self.cfg.beta_jump_limit

            # [5] 协整闸门
            is_not_cointegrated = np.isnan(adf_p[t]) or (
                        adf_p[t] > self.cfg.adf_threshold)

            # --- F. 核心信号 ---
            z = z_scores[t]

            if current_pos == 0:
                # 入场
                if not is_unstable and not is_not_cointegrated:
                    if z > self.cfg.entry_threshold:
                        current_pos = -1
                    elif z < -self.cfg.entry_threshold:
                        current_pos = 1

            elif current_pos == -1:
                # 出场
                if z < self.cfg.exit_threshold:
                    current_pos = 0
                    cooldown_counter = self.cfg.cooling_days

            elif current_pos == 1:
                # 出场
                if z > -self.cfg.exit_threshold:
                    current_pos = 0
                    cooldown_counter = self.cfg.cooling_days

            signals[t] = current_pos
            if states[t] == 0: states[t] = current_pos

        # 3. 输出
        df['signal'] = signals
        df['state'] = states
        df['vol_scalar'] = vol_scalars  # [新增] 输出仓位系数
        df['beta_jump'] = beta_jumps  # 调试用

        return df