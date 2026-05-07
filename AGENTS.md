# AeroWorld Capture Runtime Rules

1. 默认复用已经进入 PIE 的 UE 会话和 `127.0.0.1:41451` AirSim RPC，不要为了普通 episode/chunk 切换而关闭 UE 或退出 PIE。
2. 只有两类情况允许关闭 UE/PIE：需要重新编译 C++，或者用户明确要求关闭。
3. 采集脚本默认必须保持 UE 打开；需要关闭时必须显式传 `--close-ue-on-success` 或在编译前手动停止。
4. 长跑失败时默认保持 UE 打开用于检查现场、日志和调试，不要自动杀掉编辑器掩盖问题。
5. 输出目录仍必须确定性，不创建 version/timestamp 目录，失败 episode 用同一 output root 加 `--overwrite` 重跑。
6. AirSim的settings在Huawei Share下。
