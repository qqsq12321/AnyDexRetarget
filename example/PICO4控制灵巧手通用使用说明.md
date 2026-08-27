# PICO 4 控制灵巧手通用使用说明

本文给出一套通用流程：从打开 PICO、选择 `client` 或 `client2`、连接 AnyDex relay、确认安全，到启动任意已配置灵巧手的仿真、Debug 和真机控制，最后安全暂停。

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

## 1. 通用原理

```text
PICO client 或 client2
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
真实灵巧手              MuJoCo 仿真          骨架诊断
```

`client`、`client2` 和机器人型号是两个独立概念：

- `client` 或 `client2` 负责提供 PICO 手部追踪数据；
- `--robot` 决定使用哪一种灵巧手模型、配置或真机输出驱动；
- 所有本地程序统一使用 `--pico4-mode relay`；
- 不要让 `client` 和 `client2` 同时保持数据连接。

## 2. 支持范围

### 仿真和 Debug

以下 `--robot` 值可以用于 `teleop_sim.py` 和 `test/debug_skeleton.py`：

| 灵巧手 | `--robot` |
|---|---|
| Shadow Hand | `shadow` |
| Wuji Hand | `wuji` |
| Allegro Hand | `allegro` |
| LEAP Hand | `leap` |
| Inspire Hand | `inspire` |
| Ability Hand | `ability` |
| Schunk SVH | `svh` |
| RoHand | `rohand` |
| LinkerHand L21 | `linkerhand_l21` |
| Linker L20 | `linker_l20` |
| Unitree Dex5 | `unitree_dex5` |
| Sharpa Hand | `sharpa` |
| Gaia Hand20 | `gaia` |

### 真机控制

当前 `teleop_real.py` 直接实现以下真机输出：

| 灵巧手 | `--robot` | 输出方式 |
|---|---|---|
| Wuji Hand | `wuji` | Wuji 本地驱动 |
| Shadow Hand | `shadow` | TCP/ROS bridge |
| Inspire Hand | `inspire` | 串口/RS485 |
| Gaia Hand20 | `gaia` | HandSDK/SLCAN/串口 |

其他型号即使可以仿真，也不能直接用 `teleop_real.py` 控制实体硬件，除非先实现对应真机输出驱动。

## 3. 选择目标灵巧手

后续命令用 `ROBOT` 表示目标型号。例如：

```bash
ROBOT=inspire
```

切换为其他手时只修改这一行：

```bash
ROBOT=shadow
ROBOT=wuji
ROBOT=gaia
ROBOT=linker_l20
```

`ROBOT` 只对当前终端有效。仿真和 Debug 可以共用同一个值。

## 4. 打开 PICO：默认 client，按需切换 client2

### 打开 client

```bash
adb shell am force-stop com.xrobotoolkit.client2
adb shell am force-stop com.xrobotoolkit.client
adb shell am start -n \
  com.xrobotoolkit.client/com.unity3d.player.PicoVolumeKeyActivity
```

### 从 client 切换到 client2

先停止正在运行的真机控制，再执行：

```bash
adb shell am force-stop com.xrobotoolkit.client
adb shell am force-stop com.xrobotoolkit.client2
adb shell am start -n \
  com.xrobotoolkit.client2/com.unity3d.player.UnityPlayerActivity
```

### 从 client2 切回 client

先停止正在运行的真机控制，再执行：

```bash
adb shell am force-stop com.xrobotoolkit.client2
adb shell am force-stop com.xrobotoolkit.client
adb shell am start -n \
  com.xrobotoolkit.client/com.unity3d.player.PicoVolumeKeyActivity
```

每次切换后，都要在当前应用中重新确认：

1. `Network > Scan`；
2. 选择电脑 USB IP；
3. 点击 `connect`；
4. 开启 `Tracking > Hand`；
5. 开启 `Data & Control > Send`；
6. 选择 `HandTrackingActive`。

项目端不要求某种灵巧手必须绑定 `client2`。只要当前应用能把有效手部数据发送到 relay，同一套数据就可以驱动不同机器人。

## 5. 启动 relay

先检查端口：

```bash
ss -ltnp | grep -E ':(63901|63902)\b'
```

如果 `RoboticsServiceProcess` 占用 `63901`，先关闭对应软件或在其终端按 `Ctrl+C`。

打开终端 1：

```bash
cd /home/engram/AnyDexRetarget/example

/home/engram/anaconda3/envs/anydex/bin/python \
  input/pico4_daemon.py \
  --log-level DEBUG
```

成功标准：

```text
Relay hub listening on 127.0.0.1:63902
Direct server listening on 0.0.0.0:63901
Pico 4 connected from (...)
```

电脑上确认：

```bash
ss -tnp | grep 63901
```

必须看到 `ESTAB`。只有 `LISTEN` 表示 PICO 尚未连接。

## 6. 启动通用仿真

打开终端 2，先选择机器人：

```bash
cd /home/engram/AnyDexRetarget/example
ROBOT=inspire

/home/engram/anaconda3/envs/anydex/bin/python \
  teleop_sim.py \
  --input pico4 \
  --pico4-mode relay \
  --robot "$ROBOT" \
  --hand right
```

将 `ROBOT=inspire` 换成支持表中的其他值即可切换仿真手。

成功标准：

- MuJoCo 窗口打开；
- `Input FPS` 持续大于 `0`；
- PICO 右手动作能够驱动对应模型。

## 7. 启动通用 Debug

打开终端 3，选择与仿真相同的机器人：

```bash
cd /home/engram/AnyDexRetarget/example
ROBOT=inspire

/home/engram/anaconda3/envs/anydex/bin/python \
  test/debug_skeleton.py \
  --robot "$ROBOT" \
  --hand right \
  --input pico4 \
  --pico4-mode relay
```

颜色含义：

| 颜色 | 含义 |
|---|---|
| 蓝色 | 原始 PICO 手部关键点 |
| 黄色 | 使用 `pinch_scaling` 缩放后的输入 |
| 绿色 | 优化器的全手目标 |
| 红色 | 机器人 FK 重定向结果 |

## 8. 启动真机前的安全确认

必须由现场人员亲自确认：

1. 灵巧手周围已经清空；
2. 没有人的手指位于夹持区域；
3. 线缆不会被灵巧手卷入；
4. 可以立即切断驱动电源；
5. PICO 中被追踪的手保持张开和稳定；
6. 已先通过对应型号的仿真检查运动方向。

真机启动后，第一次只缓慢活动一根手指。

## 9. 不同真机的启动命令

### Inspire Hand

```bash
cd /home/engram/AnyDexRetarget/example

sg dialout -c '/home/engram/anaconda3/envs/anydex/bin/python teleop_real.py --robot inspire --input pico4 --pico4-mode relay --hand right --inspire-port /dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A10PH40A-if00-port0 --inspire-baudrate 115200 --inspire-hand-id 1'
```

### Gaia Hand20

先根据硬件确认串口、主板和 SLCAN 模式。带主板的默认示例：

```bash
cd /home/engram/AnyDexRetarget/example

/home/engram/anaconda3/envs/anydex/bin/python \
  teleop_real.py \
  --robot gaia \
  --input pico4 \
  --pico4-mode relay \
  --hand right \
  --gaia-port /dev/ttyACM0
```

无主板直连示例：

```bash
/home/engram/anaconda3/envs/anydex/bin/python \
  teleop_real.py \
  --robot gaia \
  --input pico4 \
  --pico4-mode relay \
  --hand right \
  --gaia-port /dev/ttyUSB0 \
  --gaia-baudrate 230400 \
  --no-gaia-use-slcan \
  --no-gaia-has-main-board
```

### Shadow Hand

必须先启动对应 TCP/ROS bridge：

```bash
cd /home/engram/AnyDexRetarget/example

/home/engram/anaconda3/envs/anydex/bin/python \
  teleop_real.py \
  --robot shadow \
  --input pico4 \
  --pico4-mode relay \
  --hand right \
  --docker-ip localhost \
  --docker-port 5555
```

### Wuji Hand

```bash
cd /home/engram/AnyDexRetarget/example

/home/engram/anaconda3/envs/anydex/bin/python \
  teleop_real.py \
  --robot wuji \
  --input pico4 \
  --pico4-mode relay \
  --hand right
```

启动成功时，`Input FPS` 和 `Control FPS` 都应持续大于 `0`。

## 10. 暂停和停止

必须按以下顺序：

1. 先在 `teleop_real.py` 终端按 `Ctrl+C`；
2. 确认真机驱动已经关闭；
3. 再停止 `teleop_sim.py`；
4. 再停止 `debug_skeleton.py`；
5. 需要快速恢复时保留 `pico4_daemon.py`；
6. 需要完全停止时，最后停止 `pico4_daemon.py`。

部分真机停止后会保持最后姿态，不会自动张开。姿态危险时立即断电。

## 11. 如果让我代为执行

### 准备链路，不启动真机

```text
帮我使用 PICO client，检查连接、释放端口、启动 relay，并准备 ROBOT_NAME 的仿真和 Debug，先不要启动真实手。
```

把 `ROBOT_NAME` 换成 `inspire`、`gaia`、`shadow` 等实际型号。

### 切换到 client2

```text
帮我停止当前控制，从 client 切换到 client2，重新连接 relay，先不要启动真机。
```

### 安全确认后启动真机

```text
现场已清空，可以启动 ROBOT_NAME 真实手；先验证硬件连接，再开始控制。
```

### 同时打开仿真和 Debug

```text
帮我打开 ROBOT_NAME 的仿真和 debug_skeleton，保持真实手继续运行。
```

### 暂停

```text
帮我暂停：先停止真实手，再关闭仿真和 Debug，保留 relay。
```

### 完全停止

```text
帮我全部停止：真实手、仿真、Debug 和 relay 都关闭。
```

## 12. 最短操作顺序

```text
连接硬件并打开 PICO
  -> 打开 client 或切换 client2
  -> 启动 pico4_daemon.py
  -> 确认 63901 ESTAB
  -> 选择 --robot
  -> 先运行仿真和 Debug
  -> 现场安全确认
  -> 启动对应真机驱动
  -> 暂停时先停真机
```
