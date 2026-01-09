# -*- coding: utf-8 -*-
"""
File: src/strategy/base.py
Description: 策略抽象基类 (Interface Definition)
"""
# -*- coding: utf-8 -*-
"""
File: src/strategy/base.py
Description: 
策略抽象基类 (Interface Definition) - v3.1 Decoupled
Updates:
- SignalBase.run 改为接收 indicators (模型输出)，实现数理分离。
"""
from abc import ABC, abstractmethod
import pandas as pd

class ModelBase(ABC):
    """
    数学模型基类
    职责: 计算统计指标 (Beta, Z-Score, Spread)
    """
    @abstractmethod
    def fit_predict(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        """
        :param y: Leg 1 价格
        :param x: Leg 2 价格
        :return: DataFrame (含 beta, z_score, valid 等)
        """
        pass

class SignalBase(ABC):
    """
    信号生成器基类
    职责: 纯逻辑判断 (State Machine)，不进行数学建模
    """
    @abstractmethod
    def run(self, indicators: pd.DataFrame) -> pd.DataFrame:
        """
        :param indicators: ModelBase 计算出的 DataFrame (必须含 z_score, valid)
        :return: DataFrame (含 signal, state)
        """
        pass