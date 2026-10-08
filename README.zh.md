中文 | [English](README.md)

# AnyDexRetarget

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

高精度手部姿态重定向系统。支持三种优化器（**Adaptive**、**KeyVector** 和 **DexPilot**），多种灵巧手模型和多种手部追踪输入源，可用于仿真与遥操作。

## 演示

### 仿真重定向

https://github.com/user-attachments/assets/0950b2b0-ecd4-4270-abf6-5729dc05c6cb

### Quest 3 手臂遥操作

https://github.com/user-attachments/assets/4bcac46b-a603-4c0c-9d70-83d4351c9811

### Apple Vision Pro 遥操作

https://github.com/user-attachments/assets/dccdb649-4a20-422a-979c-2b1301e8836b

### Pico 4 + Linker L20 遥操作

https://github.com/user-attachments/assets/f6d87bf8-281f-4665-9023-111c90308ce2

### Pico 4 + Gaia Hand20 遥操作

https://github.com/user-attachments/assets/e3a2432a-129f-4b76-98c7-a4834b7240ba

## 相关项目

- [vr_teleop](https://github.com/qqsq12321/vr_teleop) —— 面向**机械臂**的 VR 手部追踪遥操作项目。支持 Kinova Gen3、Realman RM65（单臂/双臂）、Piper、DexForce、Aloha 等机型的仿真与/或真机控制，并以 submodule 形式引入 AnyDexRetarget 完成灵巧手部分。需要连同手臂一起控制时用它。

## 特性

- **多灵巧手支持**：Shadow Hand、Wuji Hand、Inspire Hand、Gaia Hand20 等 13 款灵巧手开箱即用
- **三种优化器**：`adaptive`（对指感知，默认）、`vector`（关键向量匹配）和 `dexpilot`（自适应距离投影，用于稳定抓取）
- **高精度对指**：自适应优化，精确的拇指-手指接触
- **实时性能**：解析梯度 + NLopt SLSQP（~2ms/帧）
- **多输入源**：Apple Vision Pro、Meta Quest 3、Noitom PNS-G 动捕手套、笔记本摄像头（MediaPipe）、录制数据回放

## 目录

- [相关项目](#相关项目)
- [支持的机器人](#支持的机器人)
- [仓库结构](#仓库结构)
- [安装](#安装)
- [快速开始](#快速开始)
- [API 参考](#api-参考)
- [引用](#引用)
- [致谢](#致谢)
- [联系方式](#联系方式)

## 支持的机器人

配置文件按**优化器类型**和**输入源**分类：

```
example/config/
├── adaptive/          # AdaptiveOptimizerAnalytical（默认）
│   ├── mediapipe/     # 摄像头 / 视频 / 回放
│   ├── avp/           # Apple Vision Pro
│   ├── quest3/        # Meta Quest 3
│   ├── pico4/         # Pico 4
│   └── noitom/        # Noitom PNS-G 动捕手套
└── vector/            # KeyVectorOptimizer
    ├── mediapipe/
    ├── avp/
    ├── quest3/
    ├── pico4/
    └── noitom/
```

| 机器人 | `--robot` 参数 | 配置后缀 | 说明 |
|--------|----------------|----------|------|
| **Shadow Hand** | `shadow` | `shadow_hand` | Shadow Hand + MuJoCo Menagerie 模型（默认仿真目标） |
| **Wuji Hand** | `wuji` | `wuji_hand` | 无极灵巧手，5 指 / 20 自由度 |
| **Allegro Hand** | `allegro` | `allegro_hand` | Allegro Hand，4 指 / 16 自由度 |
| **Inspire Hand** | `inspire` | `inspire_hand` | 因时灵巧手，含 mimic 关节 |
| **Ability Hand** | `ability` | `ability_hand` | Ability Hand，含 mimic 关节 |
| **Leap Hand** | `leap` | `leap_hand` | Leap Hand，4 指 / 16 自由度 |
| **SVH Hand** | `svh` | `svh_hand` | Schunk SVH Hand，含 mimic 关节 |
| **LinkerHand L21** | `linkerhand_l21` | `linkerhand_l21` | LinkerHand L21 |
| **Linker L20** | `linker_l20` | `linker_l20` | DexForce Linker L20，5 指 / 21 revolute 关节，含 mimic 关节 |
| **ROHand** | `rohand` | `rohand` | ROHand |
| **Unitree Dex5** | `unitree_dex5` | `unitree_dex5_hand` | Unitree Dex5 |
| **Sharpa Hand** | `sharpa` | `sharpa_hand` | Sharpa Wave 灵巧手，5 指 / 22 DOF |
| **Gaia Hand20** | `gaia` | `gaia_hand20` | Gaia Hand20，5 指灵巧手 |

> **Noitom 配置说明：** 目前仅对 `shadow_hand`、`wuji_hand`、`inspire_hand` 进行了大致的 Noitom 参数匹配。如需精调人手与灵巧手之间的映射精度，建议运行 `debug_skeleton.py` 可视化三套骨架进行对比：**蓝色** = 原始输入、**绿色** = scaling 后的目标、**红色** = 重定向后的 FK 结果。根据骨架大小差异调整对应 YAML 配置文件中的参数（`scaling`、`segment_scaling`、`key_vectors[].scale` 等）。
>
> ```bash
> cd example
> python test/debug_skeleton.py --robot inspire --input noitom --noitom-local-ip 192.168.5.25
> ```

## 仓库结构

```text
├── anydexretarget/
│   ├── retarget.py                        # 高层统一接口
│   ├── robot.py                           # Pinocchio 机器人包装
│   ├── mediapipe.py                       # MediaPipe 坐标变换
│   └── optimizer/                         # 优化器实现
│       ├── base_optimizer.py              # 基础优化器（FK/雅可比）
│       ├── analytical_optimizer.py        # AdaptiveOptimizerAnalytical
│       ├── key_vector_optimizer.py        # KeyVectorOptimizer
│       ├── dexpilot_optimizer.py          # DexPilotOptimizer
│       ├── robot_configs.py               # 机器人 link/URDF 配置
│       └── utils.py                       # TimingStats, LPFilter, Huber 损失
├── example/
│   ├── teleop_sim.py                      # MuJoCo 仿真示例
│   ├── teleop_real.py                     # 真机控制
│   ├── input/                             # 输入设备模块
│   │   ├── cli.py                         # 共享的 --input 与各设备参数
│   │   ├── landmark_utils.py              # 共享 MediaPipe 关键点处理
│   │   ├── camera.py / video.py / ...     # 各输入设备
│   │   └── noitom.py                      # Noitom PNS-G 手套输入
│   ├── output/                            # retargeting 之后的输出处理，每个手型单独脚本
│   │   ├── real/                          # 真机驱动 (drivers_wuji.py, drivers_shadow.py, ...)
│   │   └── sim/                           # MuJoCo 仿真 qpos 映射 (mujoco_output.py)
│   ├── test/                              # 调试与可视化工具
│   │   ├── debug_skeleton.py              # 骨架对比查看器
│   │   ├── calibrate.py                   # 统一标定入口
│   │   ├── calibrate_rotation.py          # mediapipe_rotation 标定
│   │   ├── calibrate_scaling.py           # segment_scaling 标定
│   │   └── calibrate_pinch_scaling.py     # pinch_scaling 标定
│   ├── config/
│   │   ├── adaptive/                      # AdaptiveOptimizerAnalytical 配置
│   │   │   ├── avp/                       # Apple Vision Pro
│   │   │   ├── quest3/                    # Meta Quest 3
│   │   │   ├── mediapipe/                 # MediaPipe（摄像头/视频/回放）
│   │   │   ├── pico4/                     # Pico 4
│   │   │   └── noitom/                    # Noitom PNS-G 手套
│   │   ├── vector/                        # KeyVectorOptimizer 配置（目录结构相同）
│   │   └── dexpilot/                      # DexPilotOptimizer 配置（目录结构相同）
│   └── data/                              # 示例录制数据
├── assets/                                # 机器人 URDF / MuJoCo 资源
└── requirements.txt
```

## 安装

### 环境要求

- Python >= 3.10
- （可选）Apple Vision Pro + [Tracking Streamer](https://apps.apple.com/us/app/tracking-streamer/id6478969032) 应用
- （可选）Meta Quest 3 + [Hand Tracking Streamer](https://github.com/wengmister/hand-tracking-streamer) 应用
- （可选）Noitom PNS-G 动捕手套 + [Axis Studio](https://www.noitom.com.cn/axis-studio)（Windows）

### 安装步骤

```bash
# GitHub
git clone https://github.com/qqsq12321/AnyDexRetarget.git
# 或 Gitee
git clone https://gitee.com/gx_robot/AnyDexRetarget.git
cd AnyDexRetarget

# （推荐）创建并激活 conda 虚拟环境
conda create -n anydex python=3.10 -y
conda activate anydex

# 通过 conda 安装 pinocchio（推荐，预编译二进制包）
conda install -c conda-forge pinocchio

# 安装其他依赖
pip install -r requirements.txt
pip install -e .
```

### 故障排除

**macOS MuJoCo**：仿真脚本使用 `mjpython` 代替 `python`：
```bash
mjpython example/teleop_sim.py --input video --file example/data/right.mp4
```

## 快速开始

仓库当前自带两个示例输入：

- `example/data/right.mp4`：示例视频输入
- `example/data/avp1.pkl`：可选的录制回放输入

### 仿真

```bash
cd example

# 运行仓库自带示例视频（adaptive 优化器，默认）
python teleop_sim.py --input video --file data/right.mp4 --robot shadow --hand right

# Gaia Hand20（支持左右手）
python teleop_sim.py --input video --file data/right.mp4 --robot gaia --hand right

# Pico 4 直连模式（PC 广播自身地址并接收头显连接）
python teleop_sim.py --input pico4 --pico4-mode direct --robot gaia --hand right

# Pico 4 中继模式（默认；需要先在另一个终端运行 input/pico4_daemon.py）
python teleop_sim.py --input pico4 --robot gaia --hand right

# 切换到 KeyVector 优化器
python teleop_sim.py --input video --file data/right.mp4 --robot shadow --hand right --optimizer vector

# 切换到 DexPilot 优化器
python teleop_sim.py --input video --file data/right.mp4 --robot shadow --hand right --optimizer dexpilot

# 回放可选示例录制数据
python teleop_sim.py --input replay --file data/avp1.pkl --robot shadow --hand right

# 笔记本摄像头实时遥操作（MediaPipe）
python teleop_sim.py --input camera --robot shadow --hand right

# Vision Pro 实时遥操作
python teleop_sim.py --input avp --avp-ip <vision-pro-ip> --robot shadow --hand right

# Quest 3 实时遥操作（通过 Hand Tracking Streamer）
python teleop_sim.py --input quest3 --quest3-port 9000 --robot shadow --hand right

# RealSense 实时遥操作
python teleop_sim.py --input realsense --show-video --robot shadow --hand right

# Noitom PNS-G 手套
python teleop_sim.py --input noitom --noitom-local-ip 192.168.5.25 --robot inspire --hand right

# 回放你自己的录制文件（.pkl）
python teleop_sim.py --input replay --file path/to/record.pkl --robot shadow --hand right
```

### 真机控制

`teleop_real.py` 已提供 **Wuji Hand**、**Shadow Hand**（TCP 桥接）、**Inspire Hand**（串口）和 **Gaia Hand20**（官方 HandSDK）的真机输出驱动。

```bash
cd example

# Vision Pro -> Wuji Hand（adaptive）
python teleop_real.py --robot wuji --input avp --avp-ip <vision-pro-ip> --hand right

# Vision Pro -> Wuji Hand（vector 优化器）
python teleop_real.py --robot wuji --input avp --avp-ip <vision-pro-ip> --hand right --optimizer vector

# Noitom PNS-G 手套 -> Inspire Hand
python teleop_real.py --robot inspire --input noitom --hand right --noitom-local-ip 192.168.5.25

# Pico 4 中继 -> Gaia Hand20 右手
python teleop_real.py --robot gaia --input pico4 --hand right --pico4-mode relay \
  --gaia-port /dev/ttyACM0

# 回放可选示例录制数据 -> Wuji Hand
python teleop_real.py --robot wuji --input replay --file data/avp1.pkl --hand right

# 同上，使用 DexPilot 优化器
python teleop_real.py --robot wuji --input replay --file data/avp1.pkl --hand right --optimizer dexpilot

# Linux USB 权限（Inspire / Gaia 示例）
sudo chmod a+rw /dev/ttyUSB0
sudo chmod a+rw /dev/ttyACM0
```

#### Gaia Hand20 配置

安装与 Python 版本和主机架构匹配的 Gaia HandSDK wheel。推荐的 Python 3.10 Linux x86_64 环境可使用：

```bash
conda activate anydex
pip install /path/to/gaia_hand/02.HandSDK/packages/02.Linux/x86_64/v1.1.1/handsdk-1.1.1-cp310-cp310-manylinux_2_35_x86_64.whl
python -c "import hand; print('Gaia HandSDK OK')"
```


### 命令参考

以下输入参数由 `teleop_sim.py`、`teleop_real.py`、`debug_skeleton.py` 和 `calibrate_*.py` 共用（定义在 `example/input/cli.py`）。

#### 输入源

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--input` | `video`（sim）/ `replay`（real） | `avp` / `quest3` / `pico4` / `noitom` / `camera` / `realsense` / `video` / `replay` |
| `--file FILE` | `data/right.mp4`（video）/ `data/avp1.pkl`（replay） | `video` / `replay` 输入的文件 |
| `--hand` | `right` | 手的方向（`left`/`right`） |
| `--show-video` | 关闭 | 显示 RGB / 关键点预览（相机类输入） |

#### 设备连接

只有 `--input` 所选设备的参数会生效。

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--avp-ip` | `192.168.50.127` | Vision Pro IP |
| `--quest3-port` | `9000` | Quest 3 HTS 监听端口 |
| `--quest3-protocol` | `udp` | Quest 3 HTS 传输协议（`udp`/`tcp`） |
| `--pico4-mode` | `relay` | `relay`（经 `input/pico4_daemon.py` 中继）或 `direct`（PC 作 TCP 服务端） |
| `--pico4-relay-host` / `--pico4-relay-port` | `127.0.0.1` / `63902` | 中继模式：daemon 地址 |
| `--pico4-port` / `--pico4-broadcast-port` | `63901` / `29888` | 直连模式：TCP 监听端口 / UDP 广播端口 |
| `--noitom-local-ip` | `192.168.5.25` | Noitom：本机 IP |
| `--noitom-local-port` | `8000` | Noitom：本机 UDP 端口 |
| `--noitom-server-ip` | `192.168.5.33` | Noitom：Axis Studio IP（Windows） |
| `--noitom-server-port` | `9000` | Noitom：Axis Studio 端口 |

#### 回放与录制

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--speed` | `1.0` | `video` / `replay` 播放速度 |
| `--no-loop` | 关闭 | 播完即停，不循环 |
| `--record` | 关闭 | 录制输入数据（仅 teleop 脚本） |
| `--record-file FILE` | 自动生成 | 录制输出文件路径 |

#### 优化器

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--optimizer` | `adaptive` | 优化器类型：`adaptive`、`vector` 或 `dexpilot` |
| `--config` | 自动选择 | 配置文件（覆盖 `--robot` 和 `--optimizer`） |

#### 灵巧手与输出

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--robot` | `shadow`（sim）/ `wuji`（real） | 灵巧手类型；真机输出支持 `wuji`、`shadow`、`inspire` 和 `gaia` |
| `--headless` | 关闭 | 无 GUI 运行仿真 |
| `--save-sim FILE` | - | 保存离屏仿真视频 |
| `--save-qpos FILE` | - | 保存目标 / 仿真 qpos 轨迹 |

### 调试与可视化工具

#### debug_skeleton.py

在 MuJoCo 查看器中对比四套骨架，用于调试重定向问题：

- **蓝色**：原始 MediaPipe 骨架（坐标变换后，未缩放）
- **黄色**：按 `pinch_scaling` 整体缩放后的骨架
- **绿色**：缩放后的目标骨架（优化器的匹配目标）
- **红色**：机器人 FK 骨架（重定向结果）

```bash
cd example

# 摄像头输入
python test/debug_skeleton.py --robot leap --input camera

# 视频文件输入
python test/debug_skeleton.py --robot leap --input video --file data/right.mp4

# RealSense 输入
python test/debug_skeleton.py --robot shadow --input realsense

# Vision Pro 输入
python test/debug_skeleton.py --robot shadow --input avp --avp-ip <vision-pro-ip>

# 使用可选示例录制数据，对比两种优化器
python test/debug_skeleton.py --robot shadow --input replay --file data/avp1.pkl --optimizer adaptive
python test/debug_skeleton.py --robot shadow --input replay --file data/avp1.pkl --optimizer vector

# Noitom PNS-G 手套
python test/debug_skeleton.py --robot inspire --input noitom --noitom-local-ip 192.168.5.25

# Noitom + KeyVector 优化器
python test/debug_skeleton.py --robot inspire --input noitom --optimizer vector --noitom-local-ip 192.168.5.25

# RealSense D435
python test/debug_skeleton.py --robot sharpa --input realsense --hand right

# Vision Pro
python test/debug_skeleton.py --robot sharpa --input avp --avp-ip 192.168.5.32 --hand right

# 你自己的录制数据
python test/debug_skeleton.py --robot shadow --input replay --file path/to/record.pkl
```

#### adaptive contact loss

Adaptive 配置提供 `w_contact`，用于在检测到 pinch 时增加额外的指尖接触目标。该项只在 pinch alpha 非零时生效，partner 是 pinch alpha 最大的非拇指手指：

```text
L_contact = w_contact * alpha * ||thumb_tip - partner_tip||²
```

增大 `w_contact` 可以增强拇指与目标手指的指尖接触，但过大可能牺牲整体手型自然度。

#### calibrate_scaling.py

为任意灵巧手和输入源标定 `segment_scaling`。采集用户伸直手指的数据，计算机器人 FK 距离与人手距离的比值。

```bash
cd example

# RealSense 标定
python test/calibrate_scaling.py --robot sharpa --input realsense

# 视频标定
python test/calibrate_scaling.py --robot shadow --input video --file data/right.mp4

# Vision Pro 标定
python test/calibrate_scaling.py --robot wuji --input avp --avp-ip 192.168.5.32

# Noitom 标定
python test/calibrate_scaling.py --robot inspire --input noitom

# Quest 3 标定
python test/calibrate_scaling.py --robot shadow --input quest3
```

#### visualize_scaling.py

可视化 `scaling` 和 `segment_scaling` 参数对 MediaPipe 关键点的影响。

```bash
cd example

python test/visualize_scaling.py --robot leap --input video --file data/right.mp4 --hand right
python test/visualize_scaling.py --robot allegro --input replay --file data/avp1.pkl --hand right
```

## API 参考

### 基本用法

```python
from anydexretarget import Retargeter

# 从配置文件加载
retargeter = Retargeter.from_yaml("config/adaptive/mediapipe/mediapipe_shadow_hand.yaml", hand_side="right")

# 重定向：(21, 3) MediaPipe 关键点 -> 关节角度
qpos = retargeter.retarget(raw_keypoints)

# 带详细输出
qpos, info = retargeter.retarget_verbose(raw_keypoints)
print(f"Cost: {info['cost']:.4f}")
print(f"Pinch alphas: {info.get('pinch_alphas')}")  # 仅 adaptive
```

### 高级用法

```python
# 直接访问优化器
optimizer = retargeter.optimizer

# 计算给定姿态的代价
cost = optimizer.compute_cost(qpos, mediapipe_keypoints)

# 获取计时统计
stats = optimizer.get_timing_stats()
print(f"平均耗时: {stats.get_avg()['total_ms']:.2f} ms")
```

## 引用

```bibtex
@software{anydexretarget2025,
  title={AnyDexRetarget},
  author={Shiquan Qiu},
  year={2025},
  url={https://github.com/qqsq12321/AnyDexRetarget},
}
```

## 致谢

- [MuJoCo](https://mujoco.org/) - 物理仿真
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) - Shadow Hand 模型
- [dex-retargeting](https://github.com/dexsuite/dex-retargeting) - 重定向算法
- [DexPilot](https://arxiv.org/abs/1910.03135) - 基于视觉的遥操作
- [VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop) - Apple Vision Pro 数据流
- [wuji-retargeting](https://github.com/wuji-technology/wuji-retargeting) - 无极重定向

## 联系方式

如有问题，请在 [Gitee](https://gitee.com/gx_robot/AnyDexRetarget/issues) / [GitHub](https://github.com/qqsq12321/AnyDexRetarget/issues) 上提交 issue 或通过 932851972@qq.com 联系作者。
