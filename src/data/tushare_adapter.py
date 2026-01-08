# -*- coding: utf-8 -*-
"""
File: tushare_adapter.py
Created on 12/8/2025
Description:
TuShare 数据源的具体实现 (Req 1.1, 1.2, 1.4)
1. 负责与 TuShare Pro API 通信
2. 实现限频与自动重试机制 (Robustness)
3. 实现“本地配置优先”的降级策略 (Fail-over)
4. 负责将原始数据清洗为系统标准格式
包含接口限频、本地缓存、自动重试及字段标准化逻辑。
@author: zzzrrr
@version: 3.0
"""
import tushare as ts
import pandas as pd
import time
import os
import json
import hashlib
import pickle
from dotenv import load_dotenv
from .base import DataAdapter, ContractChain
def _empty_chain() -> ContractChain:
    # 每次返回新的 dict + 新的 list，避免共享引用被误改
    return {"all": [], "main": None, "submain": None, "near": None, "next": None}

load_dotenv()


class TuShareAdapter(DataAdapter):
    """
    【TuShare 适配器】
    实现 DataAdapter 定义的标准接口，底层对接 TuShare Pro API。
    """

    def __init__(self):
        token = os.getenv("TUSHARE_TOKEN")
        if not token:
            raise ValueError("错误：未在 .env 文件中找到 TUSHARE_TOKEN")

        try:
            self.pro = ts.pro_api(token)
            print("TuShare Pro API 初始化成功")
        except Exception as e:
            raise ConnectionError(f"TuShare 连接失败: {e}")

        # Req 1.4: 建立本地缓存目录
        self.cache_dir = "./data/raw/cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        # Req 1.4: 限频计数器
        self.last_req_time = 0

    def _get_cache_key(self, func_name, kwargs):
        raw_key = f"{func_name}_{sorted(kwargs.items())}"
        return hashlib.md5(raw_key.encode()).hexdigest()

    def _safe_query(self, func_name: str, use_cache=True, **kwargs):
        cache_key = self._get_cache_key(func_name, kwargs)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass

        elapsed = time.time() - self.last_req_time
        if elapsed < 0.3:
            time.sleep(0.3 - elapsed)

        for i in range(3):
            try:
                func = getattr(self.pro, func_name)
                df = func(**kwargs)
                self.last_req_time = time.time()

                if use_cache and not df.empty:
                    with open(cache_file, 'wb') as f: pickle.dump(df, f)
                return df
            except Exception:
                time.sleep(1 * (i + 1))

        raise ConnectionError(f"查询 {func_name} 失败，已重试 3 次")

    def _load_local_config(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            config_path = os.path.join(project_root, 'config',
                                       'instruments.json')

            if not os.path.exists(config_path):
                return {}
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}

    def get_contract_specs(self, symbol: str) -> pd.DataFrame:
        """
        【获取合约信息】Req 1.2 (专业分层版)
        逻辑: 优先联网 -> 失败则使用本地配置构造
        增强:
        1. 适配 static/override 分层结构
        2. 输出 DataFrame 携带 specs_as_of 和 specs_source，实现审计闭环
        """
        raw_config = self._load_local_config()

        # 提取元数据和合约体
        meta_as_of = raw_config.get("specs_as_of", "N/A")
        meta_source = raw_config.get("source", "N/A")
        contracts_config = raw_config.get("contracts", {})

        # 1. 尝试联网获取
        df = pd.DataFrame()
        try:
            df = self._safe_query('fut_basic', exchange=symbol)
        except:
            pass

        # 2. 如果联网失败，使用本地配置构造 (Fallback)
        if df.empty:
            records = []
            for code, info in contracts_config.items():
                # 匹配逻辑
                if symbol == info.get('exchange') or symbol == code:
                    # 解析分层结构
                    static = info.get('static', {})
                    override = info.get('override', {})

                    records.append({
                        'ts_code': f"{code}.{info['exchange']}",
                        'symbol': code,
                        'name': info['name'],
                        # Static (自然属性)
                        'multiplier': static.get('multiplier', 10),
                        'tick_size': static.get('tick_size', 1.0),
                        # Override (人为设定)
                        'margin_rate': override.get('margin_rate', 0.1),
                        'comm_rate': override.get('comm_rate', 0.0001),
                        'expiry_date': '20991231'
                    })

            if records:
                print(
                    f"⚠️ 警告: TuShare不可用，已降级使用本地配置 (Static+Override)")
                df_fallback = pd.DataFrame(records)
                # 【新增】透传元数据
                df_fallback['specs_as_of'] = meta_as_of
                df_fallback['specs_source'] = meta_source
                return df_fallback
            else:
                return pd.DataFrame()

        # 3. 联网成功：进行字段标准化与覆盖

        # 定义辅助函数：安全获取嵌套配置
        def get_conf_val(row_symbol, section, key, default):
            if row_symbol in contracts_config:
                return contracts_config[row_symbol].get(section, {}).get(key,
                                                                         default)
            return default

        # 覆盖 Static 属性
        df['multiplier'] = df['symbol'].apply(
            lambda x: get_conf_val(x, 'static', 'multiplier', 10))
        df['tick_size'] = df['symbol'].apply(
            lambda x: get_conf_val(x, 'static', 'tick_size', 1.0))

        # 覆盖 Override 属性
        df['margin_rate'] = df['symbol'].apply(
            lambda x: get_conf_val(x, 'override', 'margin_rate', 0.1))
        df['comm_rate'] = df['symbol'].apply(
            lambda x: get_conf_val(x, 'override', 'comm_rate', 0.0001))

        # 统一到期日
        if 'delist_date' in df.columns:
            df['expiry_date'] = df['delist_date']
        else:
            df['expiry_date'] = '20991231'

        # 【新增】透传元数据 (即使是 TuShare 数据，其规则参数也由本地配置“背书”)
        df['specs_as_of'] = meta_as_of
        df['specs_source'] = f"TuShare + {meta_source}"

        return df
    def get_daily_bars(self, contract: str, start_date: str,
                       end_date: str) -> pd.DataFrame:
        try:
            df = self._safe_query('fut_daily', ts_code=contract.upper(),
                                  start_date=start_date, end_date=end_date)
        except:
            return pd.DataFrame()

        if df.empty: return pd.DataFrame()

        required_cols = ['trade_date', 'open', 'high', 'low', 'close',
                         'settle', 'pre_settle', 'vol', 'oi']
        available_cols = [c for c in required_cols if c in df.columns]
        return df[available_cols].sort_values('trade_date').reset_index(
            drop=True)

    def get_chain(self, symbol: str, date: str) -> ContractChain:
        """
        【获取合约映射】Req 1.2
        返回: {"all": [], "main": ..., "submain": ..., "near": ..., "next": ...}
        """
        specs = self.get_contract_specs(symbol)
        if specs.empty:
            return _empty_chain()

        # 1. 筛选有效合约
        date_int = int(date)
        valid_contracts = []
        try:
            for _, row in specs.iterrows():
                if row['symbol'] != symbol: continue
                # 简单容错处理
                expiry = int(row['expiry_date']) if str(
                    row['expiry_date']).isdigit() else 20991231
                if expiry >= date_int:
                    valid_contracts.append(row['ts_code'])
        except:
            pass

        if not valid_contracts:
            return _empty_chain()

        # 按到期日排序 (字符串排序通常能反映时间顺序，如 2101 < 2102)
        sorted_contracts = sorted(valid_contracts)

        # 2. 确定主力 (Main)
        main_contract = None
        try:
            mapping_df = self._safe_query('fut_mapping', ts_code=symbol,
                                          trade_date=date)
            if not mapping_df.empty:
                main_contract = mapping_df.iloc[0]['mapping_ts_code']
        except:
            pass

        # 逻辑: 主力合约在排序列表中的"下一个"，通常即为次主力
        submain_contract = None
        if main_contract and main_contract in sorted_contracts:
            idx = sorted_contracts.index(main_contract)
            if idx + 1 < len(sorted_contracts):
                submain_contract = sorted_contracts[idx + 1]

        # 4. 确定近月(Near) / 次近月(Next)
        near_contract = sorted_contracts[0] if len(
            sorted_contracts) > 0 else None
        next_contract = sorted_contracts[1] if len(
            sorted_contracts) > 1 else None

        return {
            "all": sorted_contracts,
            "main": main_contract,
            "submain": submain_contract,
            "near": near_contract,
            "next": next_contract
        }