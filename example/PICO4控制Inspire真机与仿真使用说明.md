# PICO 4 控制 Inspire 真机、仿真与骨架诊断使用说明

本文记录在当前电脑上已经实际验证成功的操作流程：使用 PICO 4 的右手追踪数据，同时驱动 Inspire 真实灵巧手、MuJoCo 仿真窗口和 `debug_skeleton.py` 骨架诊断窗口。

## Conda 环境准备

本项目使用 `anydex` Conda 环境。当前用户已经执行过一次：

```bash
/home/engram/anaconda3/bin/conda init bash
```

该命令已将 Conda 初始化块写入 `~/.bashrc`。以后新打开的 Bash 终端不再需要手动执行：

```bash
source /home/engram/anaconda3/etc/profile.d/conda.sh
```

新终端直接执行：

```bash
conda activate anydex
cd /home/engram/AnyDexRetarget/example
```

激活成功后可以检查：

```bash
echo "$CONDA_DEFAULT_ENV"
command -v python
python --version
```

预期结果为环境名 `anydex`、Python 路径 `/home/engram/anaconda3/envs/anydex/bin/python` 和 Python `3.10.x`。

如果当前终端是在执行 `conda init bash` 之前打开的，关闭后重新打开即可；也可以只对当前终端执行一次 `source ~/.bashrc`。这里没有设置自动激活 `anydex`，因此每个新终端仍需执行 `conda activate anydex`，但不再需要先 source `conda.sh`。

## 1. 已验证环境

- 项目目录：`/home/engram/AnyDexRetarget/example`
- Python：`/home/engram/anaconda3/envs/anydex/bin/python`
- PICO 应用包名：`com.xrobotoolkit.client2`
- PICO 启动 Activity：`com.unity3d.player.UnityPlayerActivity`
- PICO 数据端口：TCP `63901`
- 本地 relay 端口：TCP `127.0.0.1:63902`
- PICO 发现端口：UDP `29888`
- Inspire 串口：`/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A10PH40A-if00-port0`
- Inspire 波特率：`115200`
- Inspire Hand ID：`1`
- 控制手：右手
- 重定向配置：`config/adaptive/pico4/pico4_inspire_hand.yaml`

建议始终使用 `/dev/serial/by-id/...` 路径，不要依赖可能随插拔变化的 `/dev/ttyUSB0`。

## 2. 安全要求

真实灵巧手启动前必须完成以下检查：

1. 清空灵巧手周围的人员、手指、线缆和坚硬物体。
2. 确保可以立即切断灵巧手驱动电源。
3. 启动控制时，先让 PICO 中被追踪的右手保持张开和稳定。
4. 首次测试只缓慢弯曲一根手指，确认运动方向后再进行完整动作。
5. 出现异常夹持、抖动或方向错误时，立即断电，不要只依赖软件停止。

`teleop_real.py` 停止后只关闭串口，不会主动让真实手张开。真实手可能保持最后一次收到的姿态。

## 3. 数据链路

```text
PICO 4 / XRoboToolkit
        |
        | TCP 63901
        v
input/pico4_daemon.py
        |
        | TCP 127.0.0.1:63902
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
teleop_real.py          teleop_sim.py       debug_skeleton.py
Inspire 真实手           MuJoCo 仿真          四色骨架诊断
```

三个程序必须使用 `--pico4-mode relay`。不要同时启动多个 `direct` 模式程序，否则会争用 `63901`。

## 4. 每次启动的完整流程

### 4.1 检查 PICO 和 Inspire 连接

```bash
adb devices -l

ls -l \
  /dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A10PH40A-if00-port0
```

PICO 状态必须是 `device`，串口链接应最终指向 `/dev/ttyUSB0` 或其他实际 USB 串口。

### 4.2 检查串口权限

当前账户已经加入 `dialout`，但旧终端可能尚未刷新组权限：

```bash
rg '^dialout:' /etc/group
id -nG
```

如果 `/etc/group` 中有 `engram`，但 `id -nG` 没有 `dialout`，可以注销后重新登录。也可以像本文后续命令一样，通过 `sg dialout -c '...'` 运行真实手程序。

不要使用 `sudo python teleop_real.py`，避免生成 root 所有的文件或继承错误的图形、Python 环境。

### 4.3 释放 `63901`

检查端口：

```bash
ss -ltnp | grep -E ':(63901|63902)\b'
```

如果看到 `/opt/apps/roboticsservice/RoboticsServiceProcess` 占用 `63901`，先回到启动它的终端按 `Ctrl+C`，或关闭对应软件。再次检查，确认 `63901` 已释放。

不要让 `RoboticsServiceProcess` 和 `pico4_daemon.py` 同时监听 `63901`。

### 4.4 启动 AnyDex relay

打开终端 1：

```bash
cd /home/engram/AnyDexRetarget/example

/home/engram/anaconda3/envs/anydex/bin/python \
  input/pico4_daemon.py \
  --log-level DEBUG
```

正常日志：

```text
Relay hub listening on 127.0.0.1:63902
Direct server listening on 0.0.0.0:63901
```

保持该终端运行。

### 4.5 XRoboToolkit 与 XRoboToolkitV2 的对应关系和切换

PICO 中安装了两个独立应用：

| PICO 中的软件 | Android 包名 | 启动 Activity |
|---|---|---|
| `XRoboToolkit` | `com.xrobotoolkit.client` | `com.unity3d.player.PicoVolumeKeyActivity` |
| `XRoboToolkitV2` | `com.xrobotoolkit.client2` | `com.unity3d.player.UnityPlayerActivity` |

本文前面的 `client` 指 `XRoboToolkit`，`client2` 指 `XRoboToolkitV2`。二者都负责提供 PICO 手部追踪数据，不要同时保持数据连接。本次 Inspire 真机控制实际验证使用的是 `XRoboToolkitV2/client2`。

#### 打开 XRoboToolkit/client

```bash
adb shell am force-stop com.xrobotoolkit.client2
adb shell am force-stop com.xrobotoolkit.client
adb shell am start -n \
  com.xrobotoolkit.client/com.unity3d.player.PicoVolumeKeyActivity
```

#### 从 XRoboToolkit/client 切换到 XRoboToolkitV2/client2

如果 `teleop_real.py` 正在运行，必须先按 `Ctrl+C` 停止真实手控制，并确认串口已经关闭。然后执行：

```bash
adb shell am force-stop com.xrobotoolkit.client
adb shell am force-stop com.xrobotoolkit.client2
adb shell am start -n \
  com.xrobotoolkit.client2/com.unity3d.player.UnityPlayerActivity
```

#### 从 XRoboToolkitV2/client2 切回 XRoboToolkit/client

同样先停止真实手控制，再执行：

```bash
adb shell am force-stop com.xrobotoolkit.client2
adb shell am force-stop com.xrobotoolkit.client
adb shell am start -n \
  com.xrobotoolkit.client/com.unity3d.player.PicoVolumeKeyActivity
```

如果 ADB 提示当前 Activity 被保留，直接戴上 PICO，手动打开目标应用。

每次打开或切换应用后，都要在当前应用内重新确认：

1. 打开 `Network`。
2. 执行 `Scan`。
3. 选择电脑的 USB IP。
4. 点击 `connect`。
5. 开启 `Tracking > Hand`。
6. 开启 `Data & Control > Send`。
7. 选择 `HandTrackingActive`。

然后在电脑检查：

```bash
ss -tnp | grep 63901
```

必须看到 PICO 到 `pico4_daemon.py` 的 `ESTAB`，才能启动真实 Inspire 手。如果没有连接，执行下一节的 USB 单播发现命令。

### 4.6 自动发送 USB 单播发现包

如果 PICO 没有自动连接，在终端 2 执行：

```bash
cd /home/engram/AnyDexRetarget/example

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

print(f"Sending discovery {pc_ip} -> {pico_ip}:29888")
deadline = time.monotonic() + 20.0
count = 0
try:
    while time.monotonic() < deadline:
        sock.sendto(packet, (pico_ip, 29888))
        count += 1
        time.sleep(0.5)
finally:
    sock.close()

print(f"Sent {count} packets")
PY
```

本次实际连接使用的地址为：

```text
PC USB IP:   192.168.250.222
PICO USB IP: 192.168.250.173
```

USB IP 可能在重新插拔或重启后变化，因此日常使用应运行上面的自动查询命令，不要永久写死地址。

daemon 成功时会输出：

```text
Pico 4 connected from ('PICO_USB_IP', 随机端口)
```

也可以检查：

```bash
ss -tnp | grep 63901
```

必须看到 `ESTAB`。只有 `LISTEN` 表示还没有收到 PICO 数据。

### 4.7 只读验证 Inspire 串口

先做不会下发运动角度的 Hand ID 读取：

```bash
cd /home/engram/AnyDexRetarget/example

sg dialout -c '/home/engram/anaconda3/envs/anydex/bin/python -c "from output.real import InspireSerialOutput; hand=InspireSerialOutput(\"/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A10PH40A-if00-port0\",115200,1); hand.close()"'
```

成功输出：

```text
Connected to Inspire hand serial ... @ 115200 baud (id=1).
Closed Inspire hand serial ...
```

完成这一步后再启动真机运动。

### 4.8 启动真实 Inspire 手

确认安全区域已经清空，让 PICO 中的右手保持张开。打开终端 3：

```bash
cd /home/engram/AnyDexRetarget/example

sg dialout -c '/home/engram/anaconda3/envs/anydex/bin/python teleop_real.py --robot inspire --input pico4 --pico4-mode relay --hand right --inspire-port /dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A10PH40A-if00-port0 --inspire-baudrate 115200 --inspire-hand-id 1'
```

成功时会看到：

```text
Connected to Inspire hand serial ... (id=1).
Starting teleoperation...
Control FPS: ... | Input FPS: ...
```

本次实际运行约为：

- `Control FPS`：`62.5`
- `Input FPS`：约 `120-150`

收到第一帧有效 PICO 右手数据后，程序才开始向真实手发送角度。

## 5. 同时打开仿真和骨架诊断

relay 已运行时，可以同时启动多个本地客户端。

### 5.1 Inspire MuJoCo 仿真

打开终端 4：

```bash
cd /home/engram/AnyDexRetarget/example

/home/engram/anaconda3/envs/anydex/bin/python \
  teleop_sim.py \
  --input pico4 \
  --pico4-mode relay \
  --robot inspire \
  --hand right
```

本次实际运行约为 `48.6 Render FPS`，PICO 输入约 `100 FPS` 以上。

### 5.2 四色骨架诊断窗口

打开终端 5：

```bash
cd /home/engram/AnyDexRetarget/example

/home/engram/anaconda3/envs/anydex/bin/python \
  test/debug_skeleton.py \
  --robot inspire \
  --hand right \
  --input pico4 \
  --pico4-mode relay
```

颜色含义：

| 颜色 | 含义 |
|---|---|
| 蓝色 | PICO 原始手部关键点经过基础坐标变换后的结果 |
| 黄色 | 使用 `pinch_scaling` 统一缩放后的输入 |
| 绿色 | 优化器的全手目标 |
| 红色 | Inspire 机器人 FK 重定向结果 |

## 6. 暂停和停止

### 6.1 暂停控制并保留 relay

按以下顺序在对应终端按 `Ctrl+C`：

1. 先停止 `teleop_real.py`，确认输出 `Closed Inspire hand serial`。
2. 再停止 `teleop_sim.py`。
3. 再停止 `debug_skeleton.py`。
4. 保持 `pico4_daemon.py` 运行，之后可以快速恢复。

真实手停止后会保持最后姿态。如果姿态危险，直接切断灵巧手电源。

### 6.2 完全停止

完成上述顺序后，最后在 relay 终端按 `Ctrl+C` 停止 `pico4_daemon.py`。

重新启动 daemon 后，PICO 可能需要重新打开 XRoboToolkit，并再次发送发现包。

### 6.3 检查是否已经停止

```bash
ps -ef | grep -E '[t]eleop_real.py|[t]eleop_sim.py|[d]ebug_skeleton.py'
fuser -v /dev/ttyUSB0
```

没有控制进程，且 `fuser` 没有显示串口占用者，表示真实手控制已经停止。

## 7. 常见问题

### `Permission denied` 打不开串口

报错示例：

```text
Permission denied: '/dev/serial/by-id/...'
```

原因是当前终端尚未获得 `dialout` 附加组。注销并重新登录，或使用本文的 `sg dialout -c '...'` 命令。

### `Pico4 relay connect failed: Connection refused`

原因是 `127.0.0.1:63902` 没有 relay 监听。先启动：

```bash
/home/engram/anaconda3/envs/anydex/bin/python \
  input/pico4_daemon.py --log-level DEBUG
```

### daemon 报 `address already in use`

检查：

```bash
ss -ltnp | grep -E ':(63901|63902)\b'
```

如果 `RoboticsServiceProcess` 占用 `63901`，先关闭它。不要重复启动第二个 daemon。

### daemon 只有 `LISTEN`，没有 `ESTAB`

说明 PICO 尚未连接：

1. 确认当前打开的是预期应用：`XRoboToolkit/client` 或 `XRoboToolkitV2/client2`；本次实测 Inspire 流程使用 `client2`。
2. 确认 `HandTrackingActive`、`Tracking > Hand` 和 `Data & Control > Send` 已开启。
3. 重启 XRoboToolkit。
4. 运行 USB 单播发现命令。

### 仿真有窗口但不动

观察终端：

- `Input FPS = 0`：先处理 PICO、relay 和 `63901 ESTAB`。
- `Input FPS > 0`：检查 `--robot inspire`、`--hand right` 和配置文件。

### 真实手不动但没有串口报错

依次确认：

1. PICO 右手处于摄像头可见范围。
2. `Input FPS` 持续大于 `0`。
3. `Control FPS` 持续大于 `0`。
4. 灵巧手驱动电源已经开启。
5. Hand ID 为 `1`，波特率为 `115200`。

## 8. 快速启动命令速查

启动顺序：

```text
1. 关闭 RoboticsServiceProcess
2. 启动 input/pico4_daemon.py
3. 打开 com.xrobotoolkit.client2
4. 确认 63901 ESTAB
5. 只读验证 Inspire Hand ID
6. 启动 teleop_real.py
7. 可选启动 teleop_sim.py
8. 可选启动 debug_skeleton.py
```

停止顺序：

```text
1. teleop_real.py
2. teleop_sim.py
3. debug_skeleton.py
4. pico4_daemon.py（需要完全停止时）
```
