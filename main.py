# -*- coding: utf-8 -*-
"""
File: main.py
Description: 量化系统 - 中央控制台
职责:
1. 作为项目的唯一启动入口
2. 引导用户选择任务 (离线冒烟 / 在线集成 / 自动化测试)
3. 调度 scripts/ 目录下的具体执行脚本

注意: 本文件不包含具体业务逻辑，业务逻辑请前往 scripts/ 或 src/
"""
import sys
import os


def main():
    print("==================================================")
    print("         Project A - 量化研究系统 (Dev)")
    print("==================================================")
    print("请选择要执行的任务:")

    print("\n[1]  离线冒烟测试 (scripts/run_smoke.py)")
    print("    -> 验证数学模型、存储逻辑")
    print("    -> 特点: 极快、不联网、不消耗 Token")

    print("\n[2]  在线集成测试 (scripts/run_live.py)")
    print("    -> 连接 TuShare，拉取真实行情，生成快照")
    print("    -> 特点: 依赖网络、测试全链路数据管道")

    print("\n[3]  运行 Pytest 自动化测试")
    print("    -> 扫描 tests/ 目录运行所有单元测试")
    print("    -> 特点: 标准化质检，显示 Pass/Fail")

    print("--------------------------------------------------")
    choice = input("请输入序号 (1/2/3) [默认: 1]: ").strip()

    # 默认选项
    if not choice:
        choice = '1'

    # 获取当前 python解释器路径 (兼容 venv)
    python_exec = sys.executable

    if choice == '1':
        print("\n>> 正在启动离线冒烟测试...")
        os.system(f'"{python_exec}" scripts/run_smoke.py')

    elif choice == '2':
        print("\n>> 正在启动在线集成测试...")
        os.system(f'"{python_exec}" scripts/run_live.py')

    elif choice == '3':
        print("\n>> 正在启动 Pytest...")
        # -v 显示详细信息
        os.system(f'"{python_exec}" -m pytest -v')

    else:
        print(" 无效输入，程序退出。")


if __name__ == "__main__":
    main()