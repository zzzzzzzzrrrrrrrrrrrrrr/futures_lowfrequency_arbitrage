# -*- coding: utf-8 -*-
"""
File: base
Created on 12/7/2025 11:34 PM

Description: 
数据适配器接口定义 (Req 1.2)
定义了策略层与数据层交互的唯一标准协议。所有具体的数据源（TuShare, LocalCSV, Wind等）
都必须继承此类并实现其抽象方法，实现依赖倒置。

@author: zzzrrr
@email: [你的邮箱]
@version: 1.0
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional, TypedDict

class ContractChain(TypedDict):
    all: List[str]
    main: Optional[str]
    submain: Optional[str]
    near: Optional[str]
    next: Optional[str]

class DataAdapter(ABC):
    """
    【协议层】
    符合 Req 1.2 的抽象基类。
    策略层只能调用这里定义的方法，不知道底层是 TuShare 还是 CSV。
    """

    @abstractmethod
    def get_contract_specs(self, symbol: str) -> pd.DataFrame:
        """获取合约规格：乘数、tick、手续费等"""
        pass

    @abstractmethod
    def get_chain(self, symbol: str, date: str) -> ContractChain:
        """返回 {"all":[], "main":..., "submain":..., "near":..., "next":...}"""
        pass

    @abstractmethod
    def get_daily_bars(self, contract: str, start_date: str,
                       end_date: str) -> pd.DataFrame:
        """
        获取日线行情
        必须包含标准字段: open, high, low, close, vol, oi (持仓量), settle (结算价)
        """
        pass