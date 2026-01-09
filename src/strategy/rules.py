"""
File: src/strategy/rules.py
Description:
策略参数配置 (v4.1 - Research Grade)
Updates:
- 新增 [协整风控] 参数: adf_threshold (判定是否均值回归)
- 新增 [预处理] 参数: use_log (是否使用对数价格)
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class StrategyConfig:
    """
    策略参数全集
    """
    # --- A. 模型参数 ---
    model_type: str = 'ols'
    window: int = 20
    z_window: int = 30
    use_log: bool = True

    # --- B. 信号阈值 (Hysteresis) ---
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5

    # --- C. 稳定性/协整闸门 ---
    beta_jump_limit: float = 0.5
    adf_threshold: float = 0.05

    # --- D. 强力风控 ---
    stop_loss: float = 4.0
    max_holding_days: int = 20

    # --- E. 仓位与杂项 ---
    # [新增] 波动率目标 (绝对值)。例如: 期望 Spread 每天波动 0.01 (1%)。
    # 仓位系数 = vol_target / spread_std
    # 如果设为 None，则不进行波动率缩放 (系数恒为 1.0)
    vol_target: Optional[float] = 0.01

    cooling_days: int = 5
    vol_filter: Optional[float] = None

    def __post_init__(self):
        if self.exit_threshold >= self.entry_threshold:
            raise ValueError("平仓阈值必须小于开仓阈值")
        if self.stop_loss <= self.entry_threshold:
            raise ValueError("止损阈值必须大于开仓阈值")