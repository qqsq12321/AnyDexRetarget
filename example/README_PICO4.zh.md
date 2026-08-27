# Pico 4 真实手部数据接入指南

本文说明如何通过 Pico 4、XRoboToolkit 和 AnyDexRetarget 获取真实手部追踪数据，并驱动 MuJoCo 中的灵巧手模型。

## 数据链路

```text
Pico 4 / XRoboToolkit
  -> USB 网络 TCP 63901
  -> input/pico4_daemon.py
  -> 本机 relay 127.0.0.1:63902
  -> teleop_sim.py
  -> MuJoCo / PKL 录制文件
```

Pico 提供每只手 26 个关节。`input/pico4.py` 会将其转换为 MediaPipe 风格的 `(21, 3)` 关键点数组：

- `left_fingers`：左手 21 个三维关键点；
- `right_fingers`：右手 21 个三维关键点。

## 1. 进入项目目录

```bash
cd /home/engram/AnyDexRetarget/example
```

本文使用项目当前环境的 Python：

```text
/home/engram/anaconda3/envs/anydex/bin/python
```

## 2. 检查 Pico USB 连接

连接 USB 数据线，并在 Pico 中允许 USB 调试，然后执行：

```bash
adb devices -l
```

设备状态必须是 `device`，不能是 `unauthorized` 或 `offline`。

查询 Pico 本次启动后的 USB IP：

```bash
adb shell ip -4 -o addr show usb0
```

Pico 每次重启或重新连接 USB 后，USB IP 都可能变化。不要长期使用上一次的 IP。

## 3. 启动 relay daemon

在终端 1 执行：

```bash
cd /home/engram/AnyDexRetarget/example

/home/engram/anaconda3/envs/anydex/bin/python \
  input/pico4_daemon.py \
  --log-level DEBUG
```

正常日志包含：

```text
Relay hub listening on 127.0.0.1:63902
Direct server listening on 0.0.0.0:63901
```

检查端口：

```bash
ss -ltnp | grep -E ':(63901|63902)\b'
```

如果端口已经由 `pico4_daemon.py` 监听，直接复用现有 daemon，不要重复启动。

## 4. 向 Pico 发送当前电脑 USB IP

先自动获取 Pico 和电脑本次连接使用的 USB IP：

```bash
PICO_USB_IP="$(
  adb shell ip -4 -o addr show usb0 \
    | tr -d '\r' \
    | awk '{print $4}' \
    | cut -d/ -f1
)"

PC_USB_IP="$(
  ip -4 route get "$PICO_USB_IP" \
    | awk '{for (i = 1; i <= NF; i++) if ($i == "src") {print $(i + 1); exit}}'
)"

echo "Pico USB IP: $PICO_USB_IP"
echo "PC USB IP:   $PC_USB_IP"
```

保持 XRoboToolkit 已打开，然后发送发现包：

```bash
export PICO_USB_IP PC_USB_IP

/home/engram/anaconda3/envs/anydex/bin/python - <<'PY'
import os
import socket
import time

from input.pico4 import _build_broadcast_packet

pico_ip = os.environ["PICO_USB_IP"]
pc_ip = os.environ["PC_USB_IP"]
packet = _build_broadcast_packet(pc_ip)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    for _ in range(30):
        sock.sendto(packet, (pico_ip, 29888))
        time.sleep(0.5)
finally:
    sock.close()

print(f"已向 Pico {pico_ip} 发送电脑 IP {pc_ip}")
PY
```

## 5. 配置 XRoboToolkit

在 Pico 头显中操作：

1. 如果出现“继续 / 退出”弹窗，先点击“继续”。
2. 打开 `Network` 页面并执行 `Scan`。
3. 选择上一步输出的 `PC USB IP`。
4. 点击 `connect`。
5. 开启 `Tracking > Hand`。
6. 开启 `Data & Control > Send`。
7. 确认交互模式为 `HandTrackingActive`。

如果 Pico 重启后列表里没有旧 IP，应重新执行第 4 节。重启后的新电脑 IP 通常与旧 IP 不同。

## 6. 验证 Pico 到电脑的数据连接

```bash
ss -tnp | grep 63901
```

成功时必须出现 `ESTAB`，连接形式类似：

```text
PC_USB_IP:63901 <-> PICO_USB_IP:随机端口
```

只有端口处于 `LISTEN` 不代表已经收到数据。没有 `ESTAB` 时，`teleop_sim.py` 得到的手部关键点会一直是全零。

## 7. 启动右手仿真

在终端 2 执行：

```bash
cd /home/engram/AnyDexRetarget/example

/home/engram/anaconda3/envs/anydex/bin/python teleop_sim.py \
  --input pico4 \
  --pico4-mode relay \
  --robot linker_l20 \
  --hand right
```

判断成功的标准：

- `Input FPS` 持续大于 `0`；
- 真实右手运动时，MuJoCo 中的右手同步运动；
- daemon 终端显示 Pico 已连接。

不要在 daemon 已运行时使用 `--pico4-mode direct`。`direct` 会与 daemon 争用 TCP `63901`。

## 8. 启动左手仿真

在另一个终端执行：

```bash
cd /home/engram/AnyDexRetarget/example

/home/engram/anaconda3/envs/anydex/bin/python teleop_sim.py \
  --input pico4 \
  --pico4-mode relay \
  --robot linker_l20 \
  --hand left
```

左右手需要两个独立的 MuJoCo 进程，但共同连接同一个 relay daemon。

## 9. 录制真实手部关键点

创建输出目录：

```bash
mkdir -p /home/engram/Documents/tmp/artifacts/pico-hand-record
```

记录右手输入：

```bash
/home/engram/anaconda3/envs/anydex/bin/python teleop_sim.py \
  --input pico4 \
  --pico4-mode relay \
  --robot linker_l20 \
  --hand right \
  --record \
  --output /home/engram/Documents/tmp/artifacts/pico-hand-record/right-input.pkl
```

录制完成后按 `Ctrl-C`。程序退出时才会将数据完整写入文件。

查看录制内容：

```bash
/home/engram/anaconda3/envs/anydex/bin/python - <<'PY'
import pickle

path = "/home/engram/Documents/tmp/artifacts/pico-hand-record/right-input.pkl"
with open(path, "rb") as file:
    frames = pickle.load(file)

print("frames:", len(frames))
if frames:
    print("time:", frames[0]["t"])
    print("left shape:", frames[0]["left_fingers"].shape)
    print("right shape:", frames[0]["right_fingers"].shape)
    print("right landmarks:\n", frames[0]["right_fingers"])
PY
```

## 10. 在电脑上显示 Pico 真实手势

仓库中的 `test/debug_skeleton.py` 可以直接读取 relay 中的真实手数据，并在 MuJoCo 窗口中显示手骨架和重定向结果。

显示右手：

```bash
cd /home/engram/AnyDexRetarget/example

/home/engram/anaconda3/envs/anydex/bin/python test/debug_skeleton.py \
  --robot linker_l20 \
  --hand right \
  --input pico4 \
  --pico4-mode relay
```

显示左手：

```bash
cd /home/engram/AnyDexRetarget/example

/home/engram/anaconda3/envs/anydex/bin/python test/debug_skeleton.py \
  --robot linker_l20 \
  --hand left \
  --input pico4 \
  --pico4-mode relay
```

窗口中的颜色含义：

- 蓝色：Pico 原始 21 点手骨架，已经过左右手坐标变换，但未做缩放；
- 黄色：应用 `pinch_scaling` 后的输入骨架；
- 绿色：应用 `segment_scaling` 后，优化器实际使用的目标骨架；
- 红色：重定向后的机器人正向运动学骨架；
- 半透明模型：Linker L20 机器人手，可用 `--alpha 0.5` 调整透明度。

如果窗口打开但没有骨架：

1. 确认 `ss -tnp | grep 63901` 中存在 `ESTAB`；
2. 确认 XRoboToolkit 已开启 `HandTrackingActive`、`Hand` 和 `Send`；
3. 将手放到 Pico 摄像头可识别的范围内；
4. 运行下一节的 Python 示例，确认对应手的数据不是全零；
5. 如果通过 SSH 使用电脑，需要有可用的图形桌面和正确的 `DISPLAY`，否则 MuJoCo 窗口无法显示。

## 11. 在 Python 中读取真实手部数据

先保持 `pico4_daemon.py` 正在运行，并确认 Pico 到电脑的 `63901` 连接为 `ESTAB`。然后在项目目录执行：

```bash
cd /home/engram/AnyDexRetarget/example

PYTHONPATH=.. /home/engram/anaconda3/envs/anydex/bin/python - <<'PY'
import time

import numpy as np

from input.pico4 import Pico4

pico = Pico4(mode="relay")

try:
    while True:
        frame = pico.get_fingers_data()
        left = frame["left_fingers"]
        right = frame["right_fingers"]

        if not np.allclose(right, 0):
            print(
                "right:", right.shape,
                "index_tip:", right[8],
                "middle_tip:", right[12],
            )
        elif not np.allclose(left, 0):
            print(
                "left:", left.shape,
                "index_tip:", left[8],
                "middle_tip:", left[12],
            )
        else:
            print("当前没有识别到手，左右手数据都是全零")

        time.sleep(0.05)
finally:
    pico.stop()
PY
```

`get_fingers_data()` 每次返回：

```python
{
    "left_fingers": np.ndarray((21, 3), dtype=np.float32),
    "right_fingers": np.ndarray((21, 3), dtype=np.float32),
}
```

本项目按 MediaPipe 风格排列 21 个关键点，并将每只手的手腕坐标归零：

| 索引 | 关键点 |
| --- | --- |
| `0` | Wrist |
| `1-4` | Thumb |
| `5-8` | Index，`8` 是食指指尖 |
| `9-12` | Middle，`12` 是中指指尖 |
| `13-16` | Ring，`16` 是无名指指尖 |
| `17-20` | Little，`20` 是小指指尖 |

XRoboToolkit 原始数据包含 26 个关节，`input/pico4.py` 会删除 Palm 和四个非拇指的 metacarpal 点，转换成上述 `(21, 3)` 格式。未识别到某只手时，该手数组为全零。

如果还要把真实关键点转换为 Linker L20 的关节目标值，可以在循环外创建 `Retargeter`，在识别到手后调用：

```python
from anydexretarget import Retargeter

retargeter = Retargeter.from_yaml(
    "config/adaptive/pico4/pico4_linker_l20.yaml",
    hand_side="right",
)

# right 的形状必须是 (21, 3)，且不能是全零。
qpos = retargeter.retarget(right)
print("Linker L20 target qpos:", qpos)
```

## 12. 保存重定向后的关节轨迹

```bash
/home/engram/anaconda3/envs/anydex/bin/python teleop_sim.py \
  --input pico4 \
  --pico4-mode relay \
  --robot linker_l20 \
  --hand right \
  --save-qpos /home/engram/Documents/tmp/artifacts/pico-hand-record/right-qpos.pkl
```

`right-qpos.pkl` 每帧包含：

- `target`：重定向后的目标关节值；
- `sim_qpos`：MuJoCo 当前关节位置；
- `sim_ctrl`：MuJoCo 控制量。

## 13. 常见故障

### XRoboToolkit 中没有电脑 IP

Pico 重启导致 USB IP 改变。重新执行第 4 节，不要继续选择旧 IP。

### `63901` 只有 `LISTEN`，没有 `ESTAB`

依次确认：

1. XRoboToolkit 没有停在“继续 / 退出”暂停弹窗；
2. 已选择当前 `PC USB IP`；
3. 已点击 `connect`；
4. `Tracking > Hand` 已开启；
5. `Data & Control > Send` 已开启。

### `Input FPS` 为 `0`

先检查：

```bash
ss -tnp | grep 63901
```

如果没有 `ESTAB`，问题发生在 Pico 到 daemon 的网络链路，不是 retarget 配置问题。

如果已有 `ESTAB`，确认双手处于 Pico 摄像头视野中，且 XRoboToolkit 显示 `HandTrackingActive`。

### 端口被占用

```bash
pgrep -af 'input/pico4_daemon.py'
ss -ltnp | grep -E ':(63901|63902)\b'
```

复用已有 daemon。不要同时运行第二个 daemon，也不要同时启动 `direct` 模式。

## 14. 每次启动的最短流程

1. USB 连接 Pico，并确认 `adb devices -l` 显示 `device`。
2. 启动或复用 `input/pico4_daemon.py`。
3. 打开 XRoboToolkit，选择 `HandTrackingActive`。
4. 动态查询 Pico/电脑 USB IP 并发送发现包。
5. 在 XRoboToolkit 中选择电脑 IP、点击 `connect`、开启 `Hand` 和 `Send`。
6. 确认 `ss -tnp | grep 63901` 显示 `ESTAB`。
7. 使用 `--pico4-mode relay` 启动仿真、骨架显示或录制。
