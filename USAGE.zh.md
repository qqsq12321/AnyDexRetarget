# AnyDexRetarget 使用方法

## 1. 激活 Conda 环境

首次在当前终端使用 Conda 时：

```bash
source /home/engram/anaconda3/etc/profile.d/conda.sh
conda activate anydex
```

确认 Python 和项目包来自 `anydex` 环境：

```bash
which python
python --version
python -c "import anydexretarget; print(anydexretarget.__file__)"
```

## 2. 运行示例仿真

进入示例目录：

```bash
cd /home/engram/AnyDexRetarget/example
```

使用仓库自带视频启动 Shadow Hand 仿真：

```bash
python teleop_sim.py \
  --video data/right.mp4 \
  --robot shadow \
  --hand right
```

无图形界面的短时验证命令：

```bash
python teleop_sim.py \
  --video data/right.mp4 \
  --robot shadow \
  --hand right \
  --headless \
  --max-frames 3 \
  --no-loop
```

查看全部参数：

```bash
python teleop_sim.py --help
python teleop_real.py --help
```

## 3. 常用输入方式

摄像头实时输入：

```bash
python teleop_sim.py --input camera --robot shadow --hand right
```

回放仓库自带的录制数据：

```bash
python teleop_sim.py --play data/avp1.pkl --robot shadow --hand right
```

使用 KeyVector 优化器：

```bash
python teleop_sim.py \
  --video data/right.mp4 \
  --robot shadow \
  --hand right \
  --optimizer vector
```

Pico 4 中继模式（默认，需要先启动中继守护进程）：

```bash
python input/pico4_daemon.py
python teleop_sim.py --input pico4 --robot linker_l20 --hand right
```

Pico 4 直连模式：

```bash
python teleop_sim.py \
  --input pico4 \
  --pico4-mode direct \
  --robot linker_l20 \
  --hand right
```

## 4. 标定配置

统一入口支持旋转、全手缩放和捏合缩放三种标定：

```bash
# 标定 Pico 4 输入旋转
python test/calibrate.py rotation --robot linker_l20 --input pico4 --hand right

# 标定全手缩放，同时正确更新 adaptive 和 vector 配置
python test/calibrate.py scaling \
  --robot linker_l20 \
  --input pico4 \
  --hand right \
  --write

# 标定 adaptive 配置的 pinch_scaling
python test/calibrate.py pinch \
  --robot linker_l20 \
  --input pico4 \
  --hand right \
  --write
```

两种优化器使用不同的缩放语义：

- Adaptive `segment_scaling` 保存腕部到 MCP 以及三段指骨的逐段比例。
- Vector `key_vectors[].scale` 保存腕部到目标关节的累计比例。

不要在两种配置之间直接复制标定值。需要预览而不修改文件时添加 `--dry-run`。

## 5. 更新后自检

从仓库根目录执行：

```bash
cd /home/engram/AnyDexRetarget

# 检查 Python 文件能否编译
python -m compileall -q anydexretarget example

# 验证 Linker L20 左右手执行器映射和 FK
python example/test/verify_linker_l20_mapping.py
```

## 6. 当前已验证的核心版本

| 组件 | 版本 |
| --- | --- |
| Python | 3.10.20 |
| AnyDexRetarget | 0.1.0（editable） |
| Pinocchio | 4.1.0 |
| NumPy | 1.26.4 |
| NLopt | 2.7.1 |
| MediaPipe | 0.10.21 |
| OpenCV Contrib | 4.11.0.86 |
| MuJoCo | 3.11.0 |

检查依赖状态：

```bash
python -m pip check
```

## 7. 版本兼容注意事项

项目代码使用 MediaPipe legacy Solutions API（`mp.solutions`）。MediaPipe
`1.x` 以及较新的部分 `0.10.x` 版本已经移除该接口，因此不要直接执行无版本
约束的依赖升级。

当前兼容组合为：

```text
numpy==1.26.4
nlopt==2.7.1
mediapipe==0.10.21
opencv-contrib-python==4.11.0.86
```

如果误升级后需要恢复：

```bash
conda install -n anydex -c conda-forge "numpy=1.26.4" -y
conda run -n anydex python -m pip install \
  "nlopt==2.7.1" \
  "mediapipe==0.10.21" \
  "opencv-contrib-python==4.11.0.86"
```

## 8. 可选硬件依赖

以下依赖未包含在基础环境中，需要使用对应设备时单独安装：

- Gaia Hand20：安装厂商提供、匹配 Python 3.10 和 Linux x86_64 的 HandSDK wheel。
- Intel RealSense：安装 `pyrealsense2`。
- Inspire Hand：安装 `pyserial`，并确保串口权限正确。
- Apple Vision Pro：安装项目所需的 `avp_stream`。
- Wuji Hand：安装厂商提供的 `wujihandpy`。

退出环境：

```bash
conda deactivate
```
