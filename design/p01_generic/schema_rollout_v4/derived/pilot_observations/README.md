# Pilot observations

真实输入来自 `aw_data/capture_archives/parent_complete_210/L4-1_v1__seed00.tar.zst`，observer 为 `u_inspect_l4_1_v1`。这里只保存 tick 5、10、15、250、255、260 的实际数组和必要绑定。`inventory.json` 保留该 observer 全部归档成员路径与缺帧情况；小窗口不替代完整 episode 覆盖。

`tick_*.npz` 包含 RGB uint8、原始 float32 米制深度、uint8 类别分割、完整 LiDAR sensor-local points、hit mask、扫描时间戳与实际报告的传感器挂载。RGB 源 PNG 的 alpha 另存为 `rgb_source_alpha`。`tick_*.json` 只保存白名单来源、时钟、坐标转换、类别映射和标定，不复制整个原始侧车。

`tick_*_association.npz` 保存真实命中点的归档扫描下标、投影像素、世界/相机坐标、采样深度和类别、轴向与径向残差以及各假设下的 0.25 米支持掩码。掩码是明确容差下的几何支持，不代表对象实例身份、已确认的可见性或生产深度语义。`geometry_metrics.json` 保存全部测量；`tick_*_view.png` 展示同刻实际四种模态。

实跑发现 tick 5、250、260 的轴向中位绝对残差约 0.020、0.016、0.016 米，tick 10、15、255 则约 10.80、10.67、9.74 米。实际报告的 LiDAR 车辆姿态不能消除后者差异。变换的代数往返精度不能证明跨模态物理对齐。所选归档的自定义 fixed-world depth 生产源码不在项目插件目录内，侧车 `DepthPerspective` 名称不能确认其径向含义；轴向/径向残差均保留，不修改原始数值来缩小误差。存在空间支持的点才适合进一步研究颜色对应，未经确认的投影不能作为实测点色。

源 UAV 高度是 AGL，capture-world Z 包含地面参考。加载结果分别保留 source pose、capture pose、capture adjustment 与实际地面参考。`source_agl_to_capture_world` 要求显式地面高度；`tangent_ground_height` 只计算 cutoff 已知点和法线的局部切平面近似。远离该点后的切平面高度是预测假设，不能冒充未来实测采集位置。`calibration_at_world_pose` 使用预测 capture-world 姿态与固定挂载，既不读取目标姿态，也不读取目标地面。

调用接口：

```python
from p01v4.data.observations import load_observation_frame

frame = load_observation_frame(
    "design/p01_generic/schema_rollout_v4/derived/pilot_observations",
    "L4-1_v1__seed00", 5, "u_inspect_l4_1_v1",
    branch="historical", cutoff=5,
)
target = load_observation_frame(
    "design/p01_generic/schema_rollout_v4/derived/pilot_observations",
    "L4-1_v1__seed00", 10, "u_inspect_l4_1_v1",
    branch="target", cutoff=5,
)
```

`frame` 返回实际数组、`calibration` 和 `binding`。所有数组只读；未来帧必须显式使用 `target` 分支，不能进入历史输入。`binding.source_entity_geometry` 是源坐标诊断，不能整体当作观测或实例分割标签传入模型。

重新生成：

```bash
PYTHONPATH=Dataset/world_model/p01_schema_rollout_v4/src \
/home/weizhiwei/data/iiot_predict/iiot_py311/airfogsim/bin/python \
  -m p01v4.data.observations \
  --archive aw_data/capture_archives/parent_complete_210/L4-1_v1__seed00.tar.zst \
  --output design/p01_generic/schema_rollout_v4/derived/pilot_observations \
  --episode L4-1_v1__seed00 --observer u_inspect_l4_1_v1 \
  --ticks 5 10 15 250 255 260 --cutoff 5
```

数值工作流适用：源归档按选定 tick 集合扫描一次，数值路径复用已索引的不可变 typed arrays；没有分布拟合、donor index 或 TRAIN/VALID/TEST 合并，未来标签与历史输入分开；加载材料化窗口无需重新扫描归档或整个数据仓库。

`episode_view.html` 展示同一 observer ticks 5–900 的 180 张真实 RGB 缩略图及四模态 payload/sidecar 覆盖；tick 0 四模态均未采集。页面无需服务端，使用同目录的 `rgb_timeline.jpg`。`episode_timeline.json` 保留各帧真实来源和执行角色，`token_regions.json` 保留名义网格 bbox 与实际 token 地址，`scan_regions.npz` 保留 262,144 个源 scan 下标、64×32 细区域、8×4 粗区域和六个选帧的实际 hit mask。页面点击区域可导出对应下标 CSV，含命中或未命中标记。

执行角色来自最终 `../current_model/result.json` 和 `bindings.json`：tick 5 是历史输入，tick 10 是预测追加，tick 15 是预测头输出，后者没有序列 token 下标。prefix 219,909 + suffix 374 + predicted append 30,627 = 250,910。tick 250/255/260 只保留归档诊断，不列为本轮模型执行。历史物化 JSON 的 branch/cutoff 不控制展示角色。

RGB 网格使用实跑的 `[1,14,28]` patch 布局，合并后为 7×14；bbox 同时保留 224×448 处理图中的精确 32×32 区域和映射到原图的连续坐标。depth/seg 是 8×16 个 90×80 原始像素区域。LiDAR 的每个粗区域包含 8,192 个真实 scan 槽位；647 维区域特征不改变该布局。RGB 和 Sonata 只称 nominal spatial support（名义空间网格），编码器注意力已混合上下文，不代表 token 的独占感受野。目标图像是实际真值对照，不能当作预测 RGB 解码图或未来输入。

分割使用固定 class ID 色表；图例标签逐项读取真实 `semantic_class_by_id`。实体栏分别保存 observer 实测传感器绑定、UAV 命令世界中心候选、landing-pad spawn 返回中心候选，以及缺少 UE actor transform、bounds、实例像素映射的对象。候选中心不能证明实例可见性；12 个 corridor 是未栅格化的逻辑区域。

导出展示：

```bash
PYTHONPATH=Dataset/world_model/p01_schema_rollout_v4/src \
/home/weizhiwei/data/iiot_predict/iiot_py311/airfogsim/bin/python \
  -m p01v4.data.observations --episode-view \
  --output design/p01_generic/schema_rollout_v4/derived/pilot_observations \
  --model-result design/p01_generic/schema_rollout_v4/derived/current_model/result.json \
  --model-bindings design/p01_generic/schema_rollout_v4/derived/current_model/bindings.json
```

已实际导出并打开检查 RGB sprite 与选帧图像，直接核对覆盖、token 地址和 scan 区域；现有 Node 完成 HTML 内脚本语法检查。本机 Playwright Chromium 可执行文件缺失，未完成浏览器交互操作，也未安装浏览器。
