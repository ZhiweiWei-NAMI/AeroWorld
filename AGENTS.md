# AeroWorld Capture Runtime Rules

1. 先读 `HANDOFF_LOW_ALTITUDE_SEMANTIC_EVENT_CHAIN.md`，并在对话框复述核心规则、入口和待检查模块，等用户批复后再继续采集或生成。
2. 默认复用已经进入 PIE 的 UE 会话和 `127.0.0.1:41451` AirSim RPC，不要为了普通 episode/chunk 切换关闭 UE 或退出 PIE。
3. 只有需要重新编译 C++，或用户明确要求关闭时，才允许关闭 UE/PIE。
4. 采集脚本默认必须保持 UE 打开；长跑失败也保持 UE 打开用于检查现场、日志和调试。
5. Python 固定使用 `E:\conda\envs\aeroagentsim\python.exe`；本环境不用 `rg`，用 PowerShell 原生命令。
6. 正式采集输出固定在 `F:\aw_cap` 和 `F:\aw_cap_summary.csv`，不创建 version/timestamp 目录。
7. 正式 episode 是 tick `0..900`，每 `5` tick 采集；UE 内存 guard 是 18GB，episode/host chunk 后必须清理 world state、临时 capture actor，并收集 PIE garbage。
8. 正式图像采集使用 UE editor-hook fixed-world camera，不使用 AirSim native camera。AirSim 只作为 RPC/兼容桥。
9. 正式 UAV 采集是单 UAV、单 channel host run；同一 episode 完成后应覆盖进入 ROI 的所有 UAV 视角，并包含 high overview。
10. AirSim settings 在 Huawei Share 下。
