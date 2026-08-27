# PICO 4 Ultra 双手驱动 AnyDexRetarget 使用指南

本文档说明如何通过 USB 将 PICO 4 Ultra 的真实双手追踪数据发送到电脑，并在 AnyDexRetarget 中驱动左右两个 Shadow Hand MuJoCo 仿真窗口，同时显示原始关节点、缩放后的目标关节点和机器人 FK 关节点。

本文档按当前机器上的实际目录、Python 环境和已验证连接方式编写：

- 项目目录：`/home/engram/AnyDexRetarget/example`
- Python：`/home/engram/anaconda3/envs/anydex/bin/python`
- PICO 应用：XRoboToolkit
- PICO ADB 包名：`com.xrobotoolkit.client`
- PICO TCP 数据端口：`63901`
- 本地 relay 端口：`63902`
- PICO 发现端口：UDP `29888`

> 重要：双手仿真必须使用 `relay` 模式。不要同时启动两个 `direct` 模式进程，因为它们会争抢同一个 TCP `63901` 端口。

## 1. 数据链路说明

完整数据链路如下：

```text
PICO 4 Ultra
  XRoboToolkit
  双手 26 关节追踪数据
        |
        | USB RNDIS 网络
        | TCP 63901
        v
input/pico4_daemon.py
  接收 PICO 数据
  26 关节转换为 MediaPipe 风格 21 关节
        |
        | 本机 TCP relay 127.0.0.1:63902
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
右手 teleop_sim.py      左手 teleop_sim.py      debug_skeleton.py
右手 MuJoCo 窗口        左手 MuJoCo 窗口        关节点诊断窗口
```

relay daemon 只与 PICO 建立一条连接，然后把同一份追踪数据转发给多个本地程序。因此可以同时运行：

- 一个右手仿真窗口；
- 一个左手仿真窗口；
- 一个或多个关节点诊断窗口。

## 2. 启动前准备

### 2.1 硬件和 PICO 端准备

1. 使用支持数据传输的 USB 线连接电脑和 PICO。
2. 戴上 PICO，允许 USB 调试授权。
3. 在 PICO 中启动 XRoboToolkit。
4. 在 XRoboToolkit 中选择手势追踪模式，确认状态为 `HandTrackingActive`。
5. 操作时保证双手在头显摄像头视野内，避免手指互相遮挡。

如果 PICO 弹出“是否允许此电脑进行 USB 调试”，应勾选“始终允许”后确认。

### 2.2 打开终端并进入项目

后续每个终端都先执行：

```bash
cd /home/engram/AnyDexRetarget/example
```

本文档直接使用 Python 的绝对路径，不依赖当前终端是否已激活 Conda 环境：

```bash
/home/engram/anaconda3/envs/anydex/bin/python
```

如果更喜欢激活 Conda，也可以执行：

```bash
source /home/engram/anaconda3/etc/profile.d/conda.sh
conda activate anydex
cd /home/engram/AnyDexRetarget/example
```

激活后，文档中的完整 Python 路径可以简写为 `python`。

## 3. 检查 USB 和 ADB

### 3.1 检查 PICO 是否被 ADB 识别

执行：

```bash
adb devices -l
```

正常输出应包含一台状态为 `device` 的设备，例如：

```text
List of devices attached
PA9410MGK8120187G  device ... model:A9210 device:sparrow
```

状态含义：

| 状态 | 含义 | 处理方法 |
|---|---|---|
| `device` | 已连接并授权 | 可以继续 |
| `unauthorized` | PICO 未授权电脑 | 戴上 PICO，确认 USB 调试弹窗 |
| 没有设备 | USB 或 ADB 未连接 | 更换数据线、USB 口，重新插拔后执行 `adb kill-server && adb start-server` |
| `offline` | ADB 通道异常 | 重新插拔 USB，必要时重启 PICO |

### 3.2 查询 PICO 的 USB IP

执行：

```bash
adb shell ip -4 -o addr show usb0
```

当前已验证的示例输出是：

```text
18: usb0 inet 192.168.219.164/24 ...
```

其中 `192.168.219.164` 是 PICO USB IP。该地址由系统分配，重新插拔或重启后可能变化，不要永久写死。

只输出 IP 可以使用：

```bash
adb shell ip -4 -o addr show usb0 \
  | tr -d '\r' \
  | awk '{print $4}' \
  | cut -d/ -f1
```

### 3.3 查询电脑对应的 USB IP

先保存 PICO IP，再查询电脑到该地址所使用的源地址：

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

echo "PICO USB IP: $PICO_USB_IP"
echo "PC USB IP:   $PC_USB_IP"
```

当前已验证示例：

```text
PICO USB IP: 192.168.219.164
PC USB IP:   192.168.219.143
```

测试 USB 网络是否能到达 PICO：

```bash
ping -c 3 "$PICO_USB_IP"
```

## 4. 启动 Pico relay daemon

打开第一个终端，执行：

```bash
cd /home/engram/AnyDexRetarget/example

/home/engram/anaconda3/envs/anydex/bin/python \
  input/pico4_daemon.py \
  --log-level DEBUG
```

daemon 的作用是：

1. 在电脑的 `0.0.0.0:63901` 等待 PICO 连接；
2. 通过 UDP `29888` 告诉 PICO 电脑的 IP；
3. 在 `127.0.0.1:63902` 接收多个本地仿真客户端；
4. 将 PICO 发送的手部追踪 JSON 转发给所有本地客户端。

正常启动时应看到类似日志：

```text
Relay hub listening on 127.0.0.1:63902
Direct server listening on 0.0.0.0:63901
```

这个终端需要一直保持运行。

### 4.1 检查端口是否监听

在另一个终端执行：

```bash
ss -ltnp | grep -E ':(63901|63902)\b'
```

正常应看到：

```text
LISTEN ... 0.0.0.0:63901   ... python
LISTEN ... 127.0.0.1:63902 ... python
```

如果出现 `address already in use`，通常表示已经有 daemon 在运行。先检查：

```bash
pgrep -af 'input/pico4_daemon.py'
ss -ltnp | grep -E ':(63901|63902)\b'
```

不要重复启动第二个 daemon。

## 5. 让 XRoboToolkit 连接 daemon

### 5.1 推荐顺序

推荐按照以下顺序操作：

1. 先启动 `input/pico4_daemon.py`；
2. 再启动或重启 PICO 中的 XRoboToolkit；
3. 在 XRoboToolkit 中选择 `HandTrackingActive`；
4. 等待 5 到 20 秒；
5. 检查 TCP `63901` 是否变成 `ESTABLISHED`。

可以通过 ADB 重启 XRoboToolkit：

```bash
adb shell am force-stop com.xrobotoolkit.client
adb shell am start -n \
  com.xrobotoolkit.client/com.unity3d.player.PicoVolumeKeyActivity
```

也可以直接在 PICO 中关闭并重新打开应用。

### 5.2 检查 XRoboToolkit 是否在前台

执行：

```bash
adb shell dumpsys window \
  | grep -E 'mCurrentFocus|mFocusedApp' \
  | tail -n 10
```

正常情况下应能看到：

```text
com.xrobotoolkit.client/com.unity3d.player.PicoVolumeKeyActivity
```

PICO 系统浮窗或快捷栏可能暂时成为当前焦点，这不一定表示 XRoboToolkit 已退出。关键是 XRoboToolkit 进程仍存在，并且 TCP 数据连接正常。

### 5.3 检查 PICO 是否已经连接电脑

执行：

```bash
ss -tnp | grep 63901
```

成功时应看到类似：

```text
ESTAB 0 0 192.168.219.143:63901 192.168.219.164:54548 ...
```

判断标准：

- 状态必须是 `ESTAB`；
- 本地端口必须是 `63901`；
- 对端地址应是 PICO 的 USB IP；
- 对应进程应是 `pico4_daemon.py` 使用的 Python 进程。

## 6. USB 自动发现失败时发送单播发现包

### 6.1 为什么可能需要单播发现

当前 `input/pico4.py` 中的 `_get_local_ips()` 依赖主机名解析。某些机器上它可能只返回以太网地址，例如 `192.168.5.5`，而没有返回 USB RNDIS 地址，例如 `192.168.219.143`。

此时 daemon 虽然已经监听 `63901`，但发出的自动广播包里不包含正确的 USB IP，PICO 就不知道应该连接哪个地址。

典型现象：

- `adb devices` 正常；
- PICO USB IP 和电脑 USB IP 正常；
- `63901` 与 `63902` 都处于 `LISTEN`；
- XRoboToolkit 已启动；
- 但 `ss -tnp | grep 63901` 始终没有 `ESTAB`。

### 6.2 自动查询 IP 并发送 20 秒单播发现包

保持 daemon 正在运行，并保持 XRoboToolkit 已启动。然后在新终端中进入项目目录：

```bash
cd /home/engram/AnyDexRetarget/example
```

执行以下完整命令：

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

export PICO_USB_IP PC_USB_IP

/home/engram/anaconda3/envs/anydex/bin/python - <<'PY'
import os
import socket
import time

from input.pico4 import _build_broadcast_packet

pico_ip = os.environ["PICO_USB_IP"]
pc_ip = os.environ["PC_USB_IP"]
port = 29888
packet = _build_broadcast_packet(pc_ip)

print(f"PICO USB IP: {pico_ip}")
print(f"PC USB IP:   {pc_ip}")
print(f"正在向 {pico_ip}:{port} 发送发现包，持续 20 秒...")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
deadline = time.monotonic() + 20.0
count = 0
try:
    while time.monotonic() < deadline:
        sock.sendto(packet, (pico_ip, port))
        count += 1
        time.sleep(0.5)
finally:
    sock.close()

print(f"完成，共发送 {count} 个发现包")
PY
```

发送期间或发送完成后检查：

```bash
ss -tnp | grep 63901
```

如果仍没有连接，可按以下顺序再试一次：

1. 保持 daemon 运行；
2. 使用 ADB 重启 XRoboToolkit；
3. 立即重新执行 20 秒单播发现命令；
4. 在 PICO 中重新选择 `HandTrackingActive`；
5. 检查 `63901` 是否进入 `ESTAB`。

## 7. 启动右手仿真

确认 PICO 到 `63901` 已经是 `ESTAB` 后，打开第二个终端：

```bash
cd /home/engram/AnyDexRetarget/example

/home/engram/anaconda3/envs/anydex/bin/python \
  teleop_sim.py \
  --input pico4 \
  --pico4-mode relay \
  --robot shadow \
  --hand right
```

正常输出包含：

```text
Robot: shadow_hand
Pico 4: relay mode 127.0.0.1:63902
Starting teleoperation...
Hand: right
Input: pico4
Render FPS: ... | Input FPS: ...
```

判断右手数据有效的关键不是只看到 `Starting teleoperation...`，而是：

- MuJoCo 右手窗口保持打开；
- `Render FPS` 持续更新；
- `Input FPS` 大于 `0`；
- 活动真实右手时，模型关节同步变化。

## 8. 启动左手仿真

打开第三个终端：

```bash
cd /home/engram/AnyDexRetarget/example

/home/engram/anaconda3/envs/anydex/bin/python \
  teleop_sim.py \
  --input pico4 \
  --pico4-mode relay \
  --robot shadow \
  --hand left
```

正常输出包含：

```text
Robot: shadow_hand
Pico 4: relay mode 127.0.0.1:63902
Starting teleoperation...
Hand: left
Input: pico4
Render FPS: ... | Input FPS: ...
```

当前程序一次只创建一只机械手模型，因此双手需要两个独立的 MuJoCo 窗口。这是正常设计，不是重复启动错误。

## 9. 启动关节点投影诊断窗口

### 9.1 左手关节点

打开第四个终端：

```bash
cd /home/engram/AnyDexRetarget/example

/home/engram/anaconda3/envs/anydex/bin/python \
  test/debug_skeleton.py \
  --robot shadow \
  --hand left \
  --input pico4 \
  --pico4-mode relay
```

### 9.2 右手关节点

```bash
cd /home/engram/AnyDexRetarget/example

/home/engram/anaconda3/envs/anydex/bin/python \
  test/debug_skeleton.py \
  --robot shadow \
  --hand right \
  --input pico4 \
  --pico4-mode relay
```

### 9.3 颜色含义

诊断窗口会显示三层骨架：

| 颜色 | 含义 | 用途 |
|---|---|---|
| 蓝色 | 原始输入经过基础坐标变换后的 21 个 PICO/MediaPipe 关节点，未做机器人尺寸缩放 | 判断 PICO 原始骨架方向、手指顺序和追踪质量 |
| 绿色 | 经过旋转、缩放和机器人配置处理后，真正送给 optimizer 的目标关节点 | 判断配置中的坐标变换和 scaling 是否合理 |
| 红色 | 重定向结果通过机器人正运动学计算得到的 FK 关节点 | 判断机械手是否跟上绿色目标骨架 |
| 半透明机械手 | MuJoCo 中的 Shadow Hand 模型 | 观察实际模型姿态 |

蓝色骨架会故意向旁边平移约 `0.15 m`，方便与绿色和红色骨架对照。因此蓝色整体不与模型重合是正常现象，重点看每根手指内部的方向和弯曲关系。

### 9.4 如何利用三种颜色定位问题

#### 蓝色已经歪或手指顺序错误

可能原因在输入层：

- PICO 手部识别不稳定；
- 手指被遮挡；
- 左右手识别错误；
- PICO 26 关节到 MediaPipe 21 关节的映射不符合数据版本；
- 原始坐标系方向异常。

#### 蓝色正常，绿色歪

可能原因在预处理或配置层：

- `rotation` 设置不正确；
- 左手镜像变换不正确；
- `scaling` 或 `segment_scaling` 不适合当前手型；
- 使用了错误的机器人或输入配置。

Shadow Hand + PICO 默认配置为：

```text
config/adaptive/pico4/pico4_shadow_hand.yaml
```

#### 绿色正常，红色明显歪

可能原因在重定向或机器人映射层：

- optimizer 对某些关节收敛不正确；
- 关节限制导致目标无法达到；
- retarget qpos 到 MuJoCo actuator 的顺序不一致；
- 左右手模型的关节正负方向不同。

#### 红色正常，但半透明模型不正常

可能原因在 MuJoCo 控制层：

- actuator 控制方向或范围不正确；
- 仿真模型的关节限制不正确；
- 控制目标与实际 qpos 之间存在较大延迟。

## 10. 一次完整启动流程

以后每次可以按这个顺序启动。

### 终端 1：relay daemon

```bash
cd /home/engram/AnyDexRetarget/example

/home/engram/anaconda3/envs/anydex/bin/python \
  input/pico4_daemon.py \
  --log-level DEBUG
```

### PICO：启动 XRoboToolkit

在 PICO 内启动 XRoboToolkit，并选择 `HandTrackingActive`。

如果需要 ADB 重启：

```bash
adb shell am force-stop com.xrobotoolkit.client
adb shell am start -n \
  com.xrobotoolkit.client/com.unity3d.player.PicoVolumeKeyActivity
```

### 检查连接

```bash
ss -tnp | grep 63901
```

必须看到 `ESTAB`。如果没有，执行第 6 节的单播发现命令。

### 终端 2：右手

```bash
cd /home/engram/AnyDexRetarget/example

/home/engram/anaconda3/envs/anydex/bin/python \
  teleop_sim.py --input pico4 --pico4-mode relay \
  --robot shadow --hand right
```

### 终端 3：左手

```bash
cd /home/engram/AnyDexRetarget/example

/home/engram/anaconda3/envs/anydex/bin/python \
  teleop_sim.py --input pico4 --pico4-mode relay \
  --robot shadow --hand left
```

### 可选终端 4：左手关节点诊断

```bash
cd /home/engram/AnyDexRetarget/example

/home/engram/anaconda3/envs/anydex/bin/python \
  test/debug_skeleton.py --robot shadow --hand left \
  --input pico4 --pico4-mode relay
```

## 11. 运行状态检查

### 11.1 检查所有相关进程

```bash
pgrep -af 'pico4_daemon.py|teleop_sim.py|debug_skeleton.py'
```

正常运行双手和左手诊断时，通常至少包含：

- 1 个 `pico4_daemon.py`；
- 1 个 `teleop_sim.py ... --hand right`；
- 1 个 `teleop_sim.py ... --hand left`；
- 可选的 `debug_skeleton.py ... --hand left`。

### 11.2 检查所有 TCP 连接

```bash
ss -tnp | grep -E '63901|63902'
```

正常结构应是：

- 1 条 PICO USB IP 到电脑 `63901` 的 `ESTAB`；
- 每个仿真或诊断程序各有 1 条到 `127.0.0.1:63902` 的 `ESTAB`。

例如同时运行左右手与一个诊断窗口时，应大致看到：

```text
PC_USB_IP:63901 <-> PICO_USB_IP:随机端口
127.0.0.1:随机端口 <-> 127.0.0.1:63902  # 右手
127.0.0.1:随机端口 <-> 127.0.0.1:63902  # 左手
127.0.0.1:随机端口 <-> 127.0.0.1:63902  # 诊断窗口
```

### 11.3 FPS 判断

`teleop_sim.py` 会周期性打印：

```text
Render FPS: 48.7 | Input FPS: 100.0
```

- `Render FPS > 0`：MuJoCo 循环正常；
- `Input FPS > 0`：该手当前有有效追踪数据；
- `Input FPS = 0` 或长时间不更新：该手没有被识别，或 relay 没有数据；
- 左右手 Input FPS 不完全相同是正常的，因为两只手可能在不同帧被识别。

当前机器实际验证时，MuJoCo 约为 `49 Render FPS`，PICO 输入约为几十到 `100 Input FPS`。具体数值会随 CPU 负载、遮挡和 PICO 追踪状态变化。

## 12. 正确停止程序

推荐按以下顺序停止：

1. 在关节点诊断终端按 `Ctrl-C`；
2. 在左右手 `teleop_sim.py` 终端分别按 `Ctrl-C`；
3. 最后在 `pico4_daemon.py` 终端按 `Ctrl-C`。

如果只是想重新启动仿真窗口，可以保持 daemon 继续运行。这样通常不需要重新连接 PICO。

如果 daemon 被停止，PICO 的 `63901` 连接会断开，重新启动 daemon 后可能需要重启 XRoboToolkit 并重新发送发现包。

如果窗口已经关闭但进程没有退出，先查询 PID：

```bash
pgrep -af 'pico4_daemon.py|teleop_sim.py|debug_skeleton.py'
```

确认目标 PID 后再停止单个进程：

```bash
kill <PID>
```

不要在未确认 PID 的情况下批量终止所有 Python 进程。

## 13. 常见问题排查

### 13.1 `adb devices` 没有设备

依次检查：

1. USB 线是否支持数据传输；
2. PICO 是否解锁；
3. 是否确认了 USB 调试授权；
4. 尝试电脑上的另一个 USB 端口；
5. 执行：

```bash
adb kill-server
adb start-server
adb devices -l
```

### 13.2 daemon 启动失败：端口被占用

检查：

```bash
ss -ltnp | grep -E ':(63901|63902)\b'
pgrep -af 'input/pico4_daemon.py'
```

如果已有 daemon 正常运行，直接复用，不要再启动一个。

### 13.3 `63901` 只有 LISTEN，没有 ESTAB

表示电脑已经等待连接，但 PICO 没有连接上。

处理顺序：

1. 确认 XRoboToolkit 正在运行；
2. 确认已选择 `HandTrackingActive`；
3. 查询当前 PICO 和电脑 USB IP；
4. 重启 XRoboToolkit；
5. 执行第 6 节的 20 秒单播发现命令；
6. 再执行 `ss -tnp | grep 63901`。

### 13.4 `63901` 已 ESTAB，但仿真 Input FPS 为 0

可能原因：

- XRoboToolkit 没有处于手势追踪模式；
- 双手不在摄像头视野内；
- 光线太暗；
- 手指或手掌被遮挡；
- 本地程序没有连到 relay `63902`。

检查本地 relay 客户端：

```bash
ss -tnp | grep 63902
```

然后把单手放到头显前方，五指张开，保持 2 到 5 秒。

### 13.5 只有一只手有数据

1. 将两手分开，避免交叉和互相遮挡；
2. 先只举起缺失的那只手；
3. 保持掌心朝向头显，五指张开；
4. 查看对应 `teleop_sim.py` 的 `Input FPS`；
5. 打开该手的 `debug_skeleton.py`，确认原始蓝色骨架是否出现。

### 13.6 左手关节歪、镜像错误或弯曲方向异常

先不要直接修改 YAML 参数，应使用三层骨架定位问题：

```bash
/home/engram/anaconda3/envs/anydex/bin/python \
  test/debug_skeleton.py \
  --robot shadow --hand left \
  --input pico4 --pico4-mode relay
```

按以下动作逐项观察：

1. 五指完全张开；
2. 握拳；
3. 只弯曲食指；
4. 只弯曲中指；
5. 拇指与食指捏合；
6. 掌心朝前和手背朝前各保持一次。

记录是哪一层开始异常：蓝色、绿色还是红色。这样才能判断需要修输入坐标、配置缩放，还是机器人关节映射。

### 13.7 MuJoCo 窗口打开但模型不动

检查：

```bash
ss -tnp | grep -E '63901|63902'
pgrep -af 'pico4_daemon.py|teleop_sim.py'
```

同时观察终端中的 `Input FPS`。如果 Input FPS 正常但模型不动，问题更可能在 retarget 或 MuJoCo 控制层；如果 Input FPS 为 0，则先处理输入链路。

### 13.8 两个仿真进程中只有一个能启动

确认命令使用的是：

```text
--pico4-mode relay
```

如果两个程序都使用 `direct`，它们会争用 `63901`。正确结构是一个 daemon 占用 `63901`，所有业务程序连接 `127.0.0.1:63902`。

### 13.9 窗口卡顿或 CPU 占用较高

每个 MuJoCo 仿真窗口是一个独立进程。当前机器上单个窗口可能占用约一个 CPU 核心，左右手加诊断窗口会进一步增加负载。

可以：

- 诊断结束后关闭 `debug_skeleton.py`；
- 只运行当前需要观察的一只手；
- 避免同时运行无关的高负载程序；
- 保持 `relay daemon` 运行，因为它本身负载较低。

## 14. 可选：保存输入和仿真结果

### 14.1 记录 Pico 输入数据

例如记录右手输入：

```bash
/home/engram/anaconda3/envs/anydex/bin/python \
  teleop_sim.py \
  --input pico4 --pico4-mode relay \
  --robot shadow --hand right \
  --record \
  --output /home/engram/Documents/tmp/artifacts/pico-record/right-input.pkl
```

保存前先创建目录：

```bash
mkdir -p /home/engram/Documents/tmp/artifacts/pico-record
```

### 14.2 保存仿真视频和 qpos

```bash
mkdir -p /home/engram/Documents/tmp/artifacts/pico-record

/home/engram/anaconda3/envs/anydex/bin/python \
  teleop_sim.py \
  --input pico4 --pico4-mode relay \
  --robot shadow --hand right \
  --save-sim /home/engram/Documents/tmp/artifacts/pico-record/right-sim.mp4 \
  --save-qpos /home/engram/Documents/tmp/artifacts/pico-record/right-qpos.pkl
```

使用 `--save-sim` 时程序会使用离屏渲染，不显示普通交互窗口。按 `Ctrl-C` 停止后，程序才会关闭并写完文件。

## 15. 端口速查表

| 端口 | 协议 | 监听方 | 连接方 | 作用 |
|---|---|---|---|---|
| `29888` | UDP | PICO/XRoboToolkit 发现端 | daemon 或单播发现脚本 | 告诉 PICO 电脑的 TCP IP |
| `63901` | TCP | `pico4_daemon.py` | PICO/XRoboToolkit | 接收 PICO 原始追踪数据 |
| `63902` | TCP，仅本机 | `pico4_daemon.py` | `teleop_sim.py`、`debug_skeleton.py` | 向多个本地程序转发追踪数据 |

## 16. 命令速查

### 检查 ADB

```bash
adb devices -l
```

### 启动 daemon

```bash
cd /home/engram/AnyDexRetarget/example
/home/engram/anaconda3/envs/anydex/bin/python input/pico4_daemon.py --log-level DEBUG
```

### 重启 XRoboToolkit

```bash
adb shell am force-stop com.xrobotoolkit.client
adb shell am start -n com.xrobotoolkit.client/com.unity3d.player.PicoVolumeKeyActivity
```

### 检查 PICO TCP 连接

```bash
ss -tnp | grep 63901
```

### 启动右手

```bash
cd /home/engram/AnyDexRetarget/example
/home/engram/anaconda3/envs/anydex/bin/python teleop_sim.py --input pico4 --pico4-mode relay --robot shadow --hand right
```

### 启动左手

```bash
cd /home/engram/AnyDexRetarget/example
/home/engram/anaconda3/envs/anydex/bin/python teleop_sim.py --input pico4 --pico4-mode relay --robot shadow --hand left
```

### 启动左手关节点诊断

```bash
cd /home/engram/AnyDexRetarget/example
/home/engram/anaconda3/envs/anydex/bin/python test/debug_skeleton.py --robot shadow --hand left --input pico4 --pico4-mode relay
```

### 检查全部进程和连接

```bash
pgrep -af 'pico4_daemon.py|teleop_sim.py|debug_skeleton.py'
ss -tnp | grep -E '63901|63902'
```

## 17. 最终成功判定

只有同时满足以下条件，才能认为双手链路正常：

1. `adb devices -l` 显示 PICO 状态为 `device`；
2. XRoboToolkit 已启动并选择 `HandTrackingActive`；
3. daemon 正在监听 TCP `63901` 和 `127.0.0.1:63902`；
4. PICO USB IP 到电脑 `63901` 的连接为 `ESTAB`；
5. 左右两个 `teleop_sim.py` 都连接到 `127.0.0.1:63902`；
6. 左右手终端的 `Input FPS` 都大于 `0`；
7. 活动真实左右手时，对应 MuJoCo 模型同步运动；
8. 打开关节点诊断时能够看到蓝色、绿色和红色骨架。

不要只凭窗口打开或出现 `Starting teleoperation...` 判断成功。必须同时检查 TCP 连接、Input FPS 和真实手部动作。
