# -*- coding: utf-8 -*-
"""
File: storage
Created on 12/8/2025 8:51 AM

Description: 
数据快照与审计模块 (Req 1.3)
1. 负责回测数据的持久化存储 (Parquet/CSV)
2. 计算数据指纹 (SHA256 Hash) 防止篡改
3. 生成审计清单 (manifest.json) 确保回测可复现

@author: zzzrrr
@email: [你的邮箱]
@version: 3.0
"""
import pandas as pd
import hashlib
import json
import os
import tushare
from datetime import datetime


class SnapshotManager:
    """
    【快照管理器】
    对应需求: Req 1.3 (数据快照与复现)
    职责:
    1. 管理回测数据的物理存储路径 (data_snapshot/)
    2. 生成数据指纹 (Hash) 防止篡改
    3. 维护审计清单 (manifest.json)
    """

    def __init__(self, base_path="./data_snapshot"):
        """
        初始化管理器
        :param base_path: 快照存储根目录。
                          根据验收标准，移出 ./data 目录，放置在项目根目录 ./data_snapshot
        """
        self.base_path = base_path

    def _calc_hash(self, df: pd.DataFrame) -> str:
        """
        【内部方法】计算 DataFrame 的 SHA256 指纹
        :param df: 需要计算的 DataFrame
        :return: 64位十六进制 Hash 字符串

        逻辑:
        先将数据转为 JSON 字符串 (orient='split') 以保证顺序和格式的一致性，
        无论在 Windows 还是 Linux 下，相同数据的 Hash 必须唯一。
        """
        # double_precision=10 保证浮点数精度在不同机器上的一致性
        data_str = df.to_json(orient='split', date_format='iso',
                              double_precision=10)
        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()

    def save_snapshot(self, data_map: dict, note: str = "",
                      extra_meta: dict = None):
        """
        【入库】保存数据快照
        自动审计逻辑: 自动从 data_map 中提取时间范围和代码，合并入 meta_info
        """
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")
        folder_path = os.path.join(self.base_path, date_str, time_str)
        os.makedirs(folder_path, exist_ok=True)

        # --- 自动审计 (Auto-Audit) ---
        auto_meta = {}
        all_dates = []
        all_codes = set()
        specs_versions = set()  # 收集静态表版本

        # 遍历所有数据表，自动抓取信息
        for name, df in data_map.items():
            if df.empty: continue

            # 1. 嗅探时间范围
            if 'trade_date' in df.columns:
                all_dates.extend(df['trade_date'].astype(str).tolist())

            # 2. 嗅探合约代码
            if 'ts_code' in df.columns:
                all_codes.update(df['ts_code'].astype(str).tolist())

            # 3. 【新增】嗅探静态数据版本
            if 'specs_as_of' in df.columns:
            # 获取该列唯一的版本号
                versions = df['specs_as_of'].astype(str).unique().tolist()
                specs_versions.update(versions)


        if all_dates:
            auto_meta['data_range_start'] = min(all_dates)
            auto_meta['data_range_end'] = max(all_dates)

        if all_codes:
            code_list = sorted(list(all_codes))
            # 避免元数据太长，只记录前10个
            auto_meta['related_symbols'] = code_list[:10]
            if len(code_list) > 10:
                auto_meta['related_symbols'].append('...')

        if specs_versions:
            auto_meta['contract_specs_version'] = list(specs_versions)
        # 合并: 用户传的 extra_meta + 自动抓的 auto_meta
        final_meta = extra_meta or {}
        final_meta.update(auto_meta)

        manifest = {
            "timestamp": now.isoformat(),
            "tushare_version": tushare.__version__,
            "note": note,
            "meta_info": final_meta,  # ✅ 这里现在包含了自动审计信息
            "files": []
        }

        print(f"[Snapshot] 正在保存快照到: {folder_path} ...")

        # 保存文件的逻辑
        for name, df in data_map.items():
            if df.empty: continue

            file_name = f"{name}.parquet"
            file_full_path = os.path.join(folder_path, file_name)

            try:
                df.to_parquet(file_full_path, index=False)
            except Exception:
                file_name = f"{name}.csv"
                file_full_path = os.path.join(folder_path, file_name)
                df.to_csv(file_full_path, index=False)

            data_hash = self._calc_hash(df)

            manifest["files"].append({
                "file": file_name,
                "rows": len(df),
                "columns": list(df.columns),
                "hash": data_hash
            })

        with open(os.path.join(folder_path, "manifest.json"), "w",
                  encoding='utf-8') as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)

        return folder_path

    def load_snapshot(self, snapshot_folder: str):
        """
        【出库】加载快照并校验完整性

        :param snapshot_folder: 快照文件夹路径
        :return: 包含 DataFrame 的字典

        逻辑:
        1. 读取 manifest.json。
        2. 逐个加载数据文件。
        3. 【核心风控】重新计算加载数据的 Hash，与清单中的 Hash 比对。
           如果不一致，说明数据被篡改或损坏，抛出严重错误阻止回测。
        """
        manifest_path = os.path.join(snapshot_folder, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"找不到清单文件: {manifest_path}")

        with open(manifest_path, "r", encoding='utf-8') as f:
            manifest = json.load(f)

        data = {}
        print(f"[Snapshot] 正在加载并校验: {snapshot_folder}")

        for file_info in manifest["files"]:
            file_path = os.path.join(snapshot_folder, file_info["file"])

            # 读取
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)

            # 校验
            current_hash = self._calc_hash(df)
            if current_hash != file_info["hash"]:
                raise ValueError(
                    f"严重警告: {file_info['file']} 数据完整性校验失败！(Hash Mismatch)")

            name = file_info["file"].split('.')[0]
            data[name] = df

        print("数据完整性校验通过")
        return data