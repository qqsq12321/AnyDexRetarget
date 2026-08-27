# Pico 4 控制左手 LinkerHand L20：V2 算法、真机映射与使用总结

## 1. 文档结论

本文记录 2026-08-27 在 AnyDexRetarget 项目中完成的 Pico 4、V2 重定向算法与左手 LinkerHand L20 真机联调结果。

当前最终结论如下：

1. Pico 4 左手 21 点数据可以稳定进入 AnyDexRetarget。
2. `AdaptiveOptimizerAnalyticalV2` 已实现独立单指跟随、唯一捏合伙伴选择和显式指尖接触距离约束。
3. 左手 L20 已通过 USB-RS485、Modbus RTU 完成真实读写和连续控制。
4. 真机最终使用 `example/output/real/drivers_l20.py` 中的 `LinkerL20V10SerialOutput`，CLI profile 为 `v10`。
5. `v10` 名称保留用于选择驱动，但其位置寄存器已经改为真机验证过的标准布局：`0-14`、`25-29` 有效，`15-24` 必须写零。
6. 四根非拇指的 `MCP roll` 已按最后一次真机反馈反向；最终实现不再对左手增加额外镜像。
7. 软件最大速度参数为 `command_hz=0`、`max_register_step=0`，实测控制循环约 `93-95 FPS`，Pico 输入约 `29-31 FPS`。
8. 本轮结束时真机控制和 MuJoCo 仿真均已停止；Pico daemon 仍在运行，但不会独立向机械手发送命令。

> 重要边界：本方案既包含通用算法，也包含 L20 专用标定。V2 优化器可以跨手型复用；L20 的 URDF/MJCF、关节命名、方向、零位、分段标定和寄存器协议不能直接复制到另一款手。

## 2. 当前文件与调用链

主要实现文件：

| 层级 | 文件 | 作用 |
|---|---|---|
| Pico 输入 | `example/input/pico4.py` | 接收并解析 Pico 手部关键点 |
| Pico 服务 | `example/input/pico4_daemon.py` | USB/网络 relay 与数据转发 |
| 真机入口 | `example/teleop_real.py` | 选择手型、V2 配置和输出驱动 |
| V2 优化器 | `anydexretarget/optimizer/analytical_optimizer_v2.py` | 单指直接控制和显式捏合约束 |
| V2 滤波 | `anydexretarget/optimizer/filter_v2.py` | 小抖动低通、大动作旁路 |
| Retargeter | `anydexretarget/retarget.py` | 坐标变换、优化、滤波和 mimic 重建 |
| L20 V2 配置 | `example/config/adaptive/pico4/pico4_linker_l20_v2.yaml` | V2 参数、左右手旋转和标定点 |
| 通用投影 | `example/output/real/joint_command_mapping.py` | 模型角到硬件命令的单调分段映射 |
| 最终 L20 驱动 | `example/output/real/drivers_l20.py` | L20 GEORT 映射、Modbus RTU 和寄存器写入 |

完整数据流为：

```text
Pico 4 左手关键点
    -> pico4_daemon relay
    -> Pico4 input，得到 21 x 3 关键点
    -> 左手坐标变换与 mediapipe_rotation
    -> AdaptiveOptimizerAnalyticalV2
    -> 独立关节自适应滤波与 mimic joint 重建
    -> L20 qpos -> 0...255 寄存器映射
    -> Modbus RTU FC10
    -> USB-RS485 / CH340
    -> 左手 LinkerHand L20
```

## 3. 硬件与通信参数

### 3.1 已确认硬件

| 项目 | 最终值 |
|---|---|
| 真机 | 左手 LinkerHand L20 |
| 转换器 | CH340 USB-RS485 |
| 稳定串口路径 | `/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0` |
| 通信协议 | Modbus RTU |
| 串口格式 | `460800 / 8N1` |
| Slave ID | `42`，十六进制 `0x2A` |
| 读位置 | `FC04`，起始地址 `0`，数量 `30` |
| 写位置 | `FC10`，起始地址 `0`，数量 `30` |
| 版本寄存器读值 | `[4, 1, 1, 10, 266, 0]` |

当前用户会话的 `dialout` 附加组可能未刷新，因此启动命令使用：

```bash
sg dialout -c 'exec ...'
```

### 3.2 Modbus RTU 帧

读输入寄存器请求体为：

$$
F_{read}=
[ID,\ 0x04,\ A_{hi},\ A_{lo},\ N_{hi},\ N_{lo},\ CRC_{lo},\ CRC_{hi}]
$$

写多个保持寄存器请求体为：

$$
F_{write}=
[ID,\ 0x10,\ A_{hi},\ A_{lo},\ N_{hi},\ N_{lo},\ 2N,\ D,\ CRC_{lo},\ CRC_{hi}]
$$

CRC16-Modbus 初值为 `0xFFFF`，生成多项式的反射形式为 `0xA001`。

## 4. V2 的通用性边界

### 4.1 通用算法部分

以下逻辑不依赖 L20 的寄存器地址或品牌：

- 人手 21 点坐标变换；
- 每根手指独立弯曲意图计算；
- 拇指横向对掌和弯曲意图计算；
- 唯一捏合伙伴选择；
- 拇指与目标指尖的显式距离损失；
- 有界 IK、Huber 鲁棒损失和解析梯度；
- 按关节动态变化量决定是否旁路低通；
- 单调分段线性模型角到执行器命令投影。

### 4.2 手型专用部分

换另一款机械手时必须重新提供或验证：

- URDF/MJCF 结构、关节轴、上下限和 mimic 关系；
- 人手语义到机械手关节的名称映射；
- 左右手坐标旋转；
- 每根手指可达的接触距离；
- 模型关节角到执行器命令的标定点；
- 正反方向、零位偏移和死区；
- CAN、RS485 或其他总线协议及寄存器布局；
- 速度、电流、使能和故障清除方式。

因此本版本不是单纯的“固定补偿方案”，也不是完全不需要标定的算法。准确说法是：

$$
\text{最终控制}=
\text{通用 V2 重定向}
+\text{手型模型}
+\text{硬件标定}
+\text{通信驱动}
$$

当前最终 `v10` 真机路径在 `drivers_l20.py` 内含 L20 专用 GEORT 标定点和零位偏移。若换手，应保留 V2 优化器，替换对应手型的模型、配置和驱动标定，不能直接沿用这些 L20 常量。

## 5. 符号约定

| 符号 | 含义 |
|---|---|
| $\mathbf p_k\in\mathbb R^3$ | Pico/MediaPipe 第 $k$ 个关键点，输入单位 m |
| $\mathbf q$ | 机械手完整模型关节向量 |
| $\mathbf q_a$ | 独立主动关节向量 |
| $\mathbf x_i(\mathbf q)$ | 第 $i$ 根机器人手指的指尖 FK 位置 |
| $J_i(\mathbf q)$ | 指尖位置对关节变量的 Jacobian |
| $d_i^h$ | 人手拇指与第 $i$ 指指尖距离 |
| $d_i^r$ | 机器人拇指与第 $i$ 指指尖距离 |
| $\alpha_i$ | 基础捏合权重 |
| $\beta$ | V2 明确接触的激活强度 |
| $\rho_\delta$ | Huber 损失 |
| $\varepsilon$ | 防止除零的小常数，约为 $10^{-8}$ |

MediaPipe 关键点索引：

| 手指 | MCP | PIP | DIP | TIP |
|---|---:|---:|---:|---:|
| 拇指 | 1 | 2 | 3 | 4 |
| 食指 | 5 | 6 | 7 | 8 |
| 中指 | 9 | 10 | 11 | 12 |
| 无名指 | 13 | 14 | 15 | 16 |
| 小指 | 17 | 18 | 19 | 20 |

腕部索引为 0。

## 6. V2 优化器数学公式

### 6.1 Huber 鲁棒损失

$$
\rho_\delta(r)=
\begin{cases}
\dfrac{1}{2}r^2, & |r|\le\delta,\\[6pt]
\delta\left(|r|-\dfrac{1}{2}\delta\right), & |r|>\delta.
\end{cases}
$$

其导数为：

$$
\rho_\delta'(r)=
\begin{cases}
r, & |r|\le\delta,\\[4pt]
\delta\operatorname{sign}(r), & |r|>\delta.
\end{cases}
$$

### 6.2 四根非拇指的独立弯曲量

对一根非拇指手指定义：

$$
\mathbf v_1=\mathbf p_{PIP}-\mathbf p_{MCP}
$$

$$
\mathbf v_2=\mathbf p_{DIP}-\mathbf p_{PIP}
$$

$$
\mathbf v_3=\mathbf p_{TIP}-\mathbf p_{DIP}
$$

两段向量夹角为：

$$
\theta(\mathbf a,\mathbf b)=
\arccos\left(
\operatorname{clip}\left(
\frac{\mathbf a^{\mathsf T}\mathbf b}
{(\lVert\mathbf a\rVert_2+\varepsilon)(\lVert\mathbf b\rVert_2+\varepsilon)},
-1,1
\right)
\right)
$$

该手指总弯曲意图为：

$$
c_i=\theta(\mathbf v_1,\mathbf v_2)+\theta(\mathbf v_2,\mathbf v_3)
$$

归一化为：

$$
r_i=\operatorname{clip}\left(
\frac{c_i-c_{open}}
{c_{closed}-c_{open}+\varepsilon},0,1
\right)
$$

再映射到该独立关节范围：

$$
q_i^{direct}=q_{i,min}+r_i(q_{i,max}-q_{i,min})
$$

L20 当前配置为：

$$
c_{open}=0.1\ \mathrm{rad},\qquad c_{closed}=2.8\ \mathrm{rad}
$$

食指、中指、无名指和小指分别独立计算，不使用四指平均值。

### 6.3 拇指横向对掌

定义掌横向向量和掌宽：

$$
\mathbf l=\mathbf p_{17}-\mathbf p_5
$$

$$
w_p=\lVert\mathbf l\rVert_2,
\qquad
\hat{\mathbf e}_{lat}=\frac{\mathbf l}{w_p+\varepsilon}
$$

拇指指尖的掌宽归一化横向位置为：

$$
h_{thumb}=
\frac{(\mathbf p_4-\mathbf p_0)^{\mathsf T}\hat{\mathbf e}_{lat}}
{w_p+\varepsilon}
$$

对掌程度为：

$$
r_{yaw}=\operatorname{clip}\left(
\frac{h_{thumb}-h_{open}}
{h_{opposed}-h_{open}+\varepsilon},0,1
\right)
$$

对应关节目标为：

$$
q_{thumb,yaw}^{direct}=
q_{yaw,min}+r_{yaw}(q_{yaw,max}-q_{yaw,min})
$$

当前参数为：

$$
h_{open}=-2.05,\qquad h_{opposed}=0.1
$$

### 6.4 拇指弯曲

拇指使用关键点 1、2、3、4 的两段夹角和：

$$
c_{thumb}=
\theta(\mathbf p_2-\mathbf p_1,\mathbf p_3-\mathbf p_2)
+
\theta(\mathbf p_3-\mathbf p_2,\mathbf p_4-\mathbf p_3)
$$

$$
r_{thumb}=\operatorname{clip}\left(
\frac{c_{thumb}-c_{thumb,open}}
{c_{thumb,closed}-c_{thumb,open}+\varepsilon},0,1
\right)
$$

当前参数为：

$$
c_{thumb,open}=0.13\ \mathrm{rad},\qquad
c_{thumb,closed}=1.69\ \mathrm{rad}
$$

该比例同时映射到配置的 `thumb_cmc_pitch` 和 `thumb_mcp` 独立轴。

### 6.5 基础捏合权重

对非拇指手指 $i$，人手指尖距离为：

$$
d_i^h=100\left\|\mathbf p_{i,tip}-\mathbf p_{thumb,tip}\right\|_2
$$

单位转换后为 cm。两级阈值生成连续捏合权重：

$$
\alpha_i=\operatorname{clip}\left(
\frac{d_{2,i}-d_i^h}
{d_{2,i}-d_{1,i}+\varepsilon},
0,\alpha_{max}
\right)
$$

当前 L20 参数为：

$$
d_1=2.0\ \mathrm{cm},\qquad
d_2=4.0\ \mathrm{cm},\qquad
\alpha_{max}=1.0
$$

### 6.6 唯一捏合伙伴

候选集合为：

$$
\mathcal F=\{index,middle,ring,pinky\}
$$

最大候选为：

$$
i^*=\underset{i\in\mathcal F}{\arg\max}\ \alpha_i
$$

若第二候选 $i^{(2)}$ 也已达到激活阈值，并且两者差值不够大，则拒绝建立接触：

$$
\alpha_{i^{(2)}}\ge\beta_{min}
\quad\land\quad
\alpha_{i^*}-\alpha_{i^{(2)}}<m_{dom}
\Longrightarrow
\text{no contact partner}
$$

当前参数为：

$$
\beta_{min}=0.8,\qquad m_{dom}=0.15
$$

这样可以防止多根手指同时靠近拇指时，优化器在不同伙伴之间来回跳变。

### 6.7 接触激活强度与目标距离

先计算：

$$
\beta_{raw}=\operatorname{clip}\left(
\frac{\alpha_{i^*}}{\alpha_{max}+\varepsilon},0,1
\right)
$$

只有 $\beta_{raw}\ge\beta_{min}$ 才进入接触状态。重新归一化后：

$$
\beta=\operatorname{clip}\left(
\frac{\beta_{raw}-\beta_{min}}
{1-\beta_{min}+\varepsilon},0,1
\right)
$$

缩放后的人手目标距离为：

$$
d_{scaled}=\max(s_{pinch}d_{i^*}^h,d_{reach})
$$

其中当前：

$$
s_{pinch}=1.5101,\qquad d_{reach}=0.2\ \mathrm{cm}
$$

最终接触目标距离从人手缩放距离连续过渡到机械手可达距离：

$$
d^*=(1-\beta)d_{scaled}+\beta d_{reach}
$$

### 6.8 显式指尖接触损失

机器人拇指与伙伴指尖距离为：

$$
d^r=\left\|\mathbf x_{i^*}(\mathbf q)-\mathbf x_{thumb}(\mathbf q)\right\|_2
$$

残差为：

$$
e_c=d^r-d^*
$$

V2 新增接触损失：

$$
L_{contact}=w_c\beta\rho_{\delta_c}(e_c)
$$

当前参数为：

$$
w_c=160,\qquad \delta_c=0.5\ \mathrm{cm}
$$

令：

$$
\mathbf r=\mathbf x_{i^*}-\mathbf x_{thumb},
\qquad
\hat{\mathbf r}=\frac{\mathbf r}{\lVert\mathbf r\rVert_2+\varepsilon}
$$

则解析梯度为：

$$
\nabla_{\mathbf q}L_{contact}=
w_c\beta\rho_{\delta_c}'(e_c)
\hat{\mathbf r}^{\mathsf T}
(J_{i^*}-J_{thumb})
$$

### 6.9 总优化目标

V2 捏合阶段继承 V1 的指尖位置、指尖方向和全手姿态目标，并增加接触项。可概括为：

$$
L(\mathbf q)=
\sum_i
\left[
\alpha_i
\left(
w_{pos}\rho_{\delta_p}(e_{pos,i})
+w_{dir}\rho_{\delta_d}(e_{dir,i})
\right)
+(1-\alpha_i)w_{full}L_{full,i}
\right]
+L_{contact}
+\lambda\lVert\mathbf q-\mathbf q_{t-1}\rVert_2^2
$$

其中全手姿态项为 PIP、DIP、TIP 三组腕部相对向量误差的平均：

$$
L_{full,i}=\frac{1}{3}
\left[
\rho_{\delta_p}(e_{pip,i})
+\rho_{\delta_p}(e_{dip,i})
+\rho_{\delta_p}(e_{tip,i})
\right]
$$

当前主要权重为：

$$
w_{pos}=5.0,\qquad
w_{dir}=1.0,\qquad
w_{full}=1.0,\qquad
\lambda=0.04
$$

优化变量受机械手关节上下界约束：

$$
\mathbf q_{min}\le\mathbf q\le\mathbf q_{max}
$$

### 6.10 直接控制与 IK 的混合

对一个配置了直接控制的关节：

$$
q^{out}=(1-\eta)q^{ik}+\eta q^{direct}
$$

L20 当前配置行为为：

- 非捏合时，四指 flexion 和拇指 yaw/pitch 的 $\eta=1$，单指直接跟随；
- 捏合时，拇指与唯一伙伴的 $\eta=0$，由 IK 和接触距离约束决定；
- 捏合时，其他三指仍保持 $\eta=1$ 的独立控制；
- `direct_only: false`，因此仍计算完整 IK，未配置为直接控制的轴继续保留整体姿态信息。

## 7. V2 自适应低通公式

对独立关节 $j$，阈值为：

$$
T_j=\max(q_{j,max}-q_{j,min},\varepsilon)r_b
$$

当前：

$$
r_b=0.006
$$

令本帧输入为 $x_{t,j}$，上一帧输出为 $y_{t-1,j}$：

$$
\Delta_{t,j}=x_{t,j}-y_{t-1,j}
$$

输出为：

$$
y_{t,j}=
\begin{cases}
x_{t,j}, & |\Delta_{t,j}|\ge T_j\ \text{或该关节为 passthrough},\\[4pt]
y_{t-1,j}+\alpha_f\Delta_{t,j}, & |\Delta_{t,j}|<T_j.
\end{cases}
$$

当前：

$$
\alpha_f=0.65
$$

以下拇指关节为 passthrough，不经过 EMA 延迟：

```text
thumb_cmc_yaw
thumb_cmc_roll
thumb_cmc_pitch
thumb_mcp
```

滤波只作用于独立关节，之后再重建 mimic joint，避免主从关节被分别滤波后破坏机械约束。

## 8. 通用单调分段投影

对于任意手型的一个模型关节，给定严格递增的输入标定点：

$$
x_0<x_1<\cdots<x_n
$$

以及单调递增或单调递减的硬件输出点：

$$
y_0,y_1,\ldots,y_n
$$

当 $x\in[x_k,x_{k+1}]$ 时：

$$
f(x)=y_k+
\frac{x-x_k}{x_{k+1}-x_k}
(y_{k+1}-y_k)
$$

超出范围时夹紧到端点：

$$
f(x)=
\begin{cases}
y_0, & x\le x_0,\\[4pt]
y_n, & x\ge x_n.
\end{cases}
$$

这一层的通用性来自“算法只要求单调标定点”，而不是写死某款手的常数。配置还支持：

```text
default -> 左右手共享
left    -> 左手覆盖
right   -> 右手覆盖
```

### 8.1 L20 配置中的三组拇指标定

`pico4_linker_l20_v2.yaml` 当前提供：

| 关节 | 模型角输入 rad | 寄存器输出 |
|---|---|---|
| `thumb_cmc_roll` | `[0.139626, 0.729907, 1.44236]` | `[255, 128, 0]` |
| `thumb_cmc_pitch` | `[0.05236, 0.449907, 0.88236]` | `[255, 128, 0]` |
| `thumb_mcp` | `[0.139626, 0.67736, 1.25]` | `[255, 128, 0]` |

注意：通用 `joint_command_mapping.py` 当前由 standard RS485/CAN 路径读取。用户最终选择的 `v10` 路径则在 `drivers_l20.py` 内使用等价的 L20 GEORT 三点归一化标定。两者不应同时叠加。

## 9. 最终 L20 真机映射

### 9.1 普通线性关节

对于没有 GEORT 三点标定的 L20 关节：

$$
n=\operatorname{clip}\left(
\frac{q+o-q_{min}}{q_{max}-q_{min}},0,1
\right)
$$

其中 $o$ 是 L20 专用零位偏移。若协议方向需要反向：

$$
n'=1-n
$$

最终 8 位寄存器为：

$$
r=\operatorname{clip}\left(\operatorname{round}(255n'),0,255\right)
$$

当前四根非拇指 `MCP pitch` 的 L20 专用偏移为：

$$
o_{mcp\_pitch}=-0.15235988\ \mathrm{rad}
$$

该偏移属于 L20 驱动标定，不属于通用 V2 算法。

### 9.2 GEORT 拇指分段映射

以 `thumb_cmc_roll` 为例，内部先映射到归一化点：

$$
(0.139626,0),\quad(0.729907,0.5),\quad(1.442360,1)
$$

该关节随后按 Modbus 方向反向：

$$
r=\operatorname{round}(255(1-n))
$$

实测捏合帧中：

$$
q_{thumb\_cmc\_roll}=0.812732\ \mathrm{rad}
$$

分段插值得：

$$
n\approx
0.5+
\frac{0.812732-0.729907}{1.442360-0.729907}\times0.5
\approx0.5581
$$

因此：

$$
r\approx255(1-0.5581)\approx112.7\approx113
$$

旧映射在约 `0.683 rad` 已提前饱和为寄存器 `0`，导致仿真中的拇指仍可继续运动，而真机已经没有剩余命令空间。这是“仿真已经对指、真机仍有明显误差”的主要根因之一。

### 9.3 最终 30 个位置寄存器布局

| 地址 | 内容 |
|---:|---|
| 0 | `thumb_cmc_roll` |
| 1-4 | 四指 roll 中性值 `128` |
| 5 | `thumb_cmc_yaw` |
| 6 | `index_mcp_roll` |
| 7 | `middle_mcp_roll` |
| 8 | `ring_mcp_roll` |
| 9 | `pinky_mcp_roll` |
| 10 | `thumb_cmc_pitch` |
| 11 | `index_mcp_pitch` |
| 12 | `middle_mcp_pitch` |
| 13 | `ring_mcp_pitch` |
| 14 | `pinky_mcp_pitch` |
| 15-24 | 保留，必须写 `0` |
| 25 | `thumb_mcp` |
| 26 | `index_pip` |
| 27 | `middle_pip` |
| 28 | `ring_pip` |
| 29 | `pinky_pip` |

严禁恢复旧版把 distal 数据重复写到 `15-29` 的布局。真机验证过的有效 distal 区只有 `25-29`。

### 9.4 四指 MCP roll 最终方向

最终实现中，以下四个关节不在 `_L20_INVERTED_JOINTS` 中：

```text
INDEX_MCP_ROLL
MIDDLE_MCP_ROLL
RING_MCP_ROLL
PINKY_MCP_ROLL
```

其线性方向为：

$$
q=-0.23\ \mathrm{rad}\Rightarrow r=0
$$

$$
q=0.23\ \mathrm{rad}\Rightarrow r=255
$$

左手与右手在驱动层使用相同的 GEORT/寄存器方向；左手不再额外乘负号。这一结果相对上一版左手镜像实现完成了反向。

## 10. 真实捏合样本与误差分析

捕获到的有效左手拇指-食指捏合样本：

```text
人体拇指-食指距离：2.165 mm
index_mcp_roll=-0.230000
index_mcp_pitch=1.220000
index_pip=0.817605
thumb_cmc_roll=0.812732
thumb_cmc_yaw=0.668937
thumb_cmc_pitch=0.012719
thumb_mcp=0.922894
```

仿真正常而真机存在误差，不能只归因于 IK。完整误差链为：

$$
e_{total}=
e_{tracking}
+e_{coordinate}
+e_{kinematics}
+e_{calibration}
+e_{actuator}
+e_{latency}
$$

其中：

- $e_{tracking}$：Pico 手部关键点噪声或遮挡；
- $e_{coordinate}$：左右手坐标系、轴方向和旋转误差；
- $e_{kinematics}$：URDF/MJCF 与实物连杆尺寸、轴位置不完全一致；
- $e_{calibration}$：模型角与 `0-255` 执行器位置映射误差；
- $e_{actuator}$：机械限位、齿隙、柔性、死区和负载误差；
- $e_{latency}$：Pico 帧率、滤波和串口命令更新延迟。

本次已明确修复的主要问题包括：

1. 拇指模型角到真机寄存器提前饱和；
2. 位置寄存器 distal 布局错误；
3. 左手四指 `MCP roll` 额外镜像方向错误；
4. 固定低速和逐寄存器渐变导致的明显跟手延迟。

仍需真机动作样本进一步标定的部分包括：

- 无名指和小指与拇指的精确对指误差；
- 拇指独立运动的所有姿态区间；
- 负载、速度变化下的回差与死区；
- L20 实物与模型指尖位置的系统误差。

## 11. 三套启动方式

以下命令均使用左手、Pico relay、V2 算法、`drivers_l20.py` 的 `v10` profile、稳定串口路径、`460800` 和 Slave ID `42`。

### 11.1 安全低速验证

适合首次通电、重新接线、重新标定或只验证一个捏合动作：

```bash
cd /home/engram/AnyDexRetarget/example

sg dialout -c 'exec /home/engram/anaconda3/envs/anydex/bin/python teleop_real.py \
  --robot linker_l20 \
  --input pico4 \
  --pico4-mode relay \
  --hand left \
  --retarget-version v2 \
  --linker-l20-transport rs485 \
  --linker-l20-port /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0 \
  --linker-l20-baudrate 460800 \
  --linker-l20-slave-id 42 \
  --linker-l20-rs485-profile v10 \
  --linker-l20-command-hz 30 \
  --linker-l20-max-register-step 1'
```

### 11.2 正常控制

兼顾响应速度与渐变保护：

```bash
cd /home/engram/AnyDexRetarget/example

sg dialout -c 'exec /home/engram/anaconda3/envs/anydex/bin/python teleop_real.py \
  --robot linker_l20 \
  --input pico4 \
  --pico4-mode relay \
  --hand left \
  --retarget-version v2 \
  --linker-l20-transport rs485 \
  --linker-l20-port /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0 \
  --linker-l20-baudrate 460800 \
  --linker-l20-slave-id 42 \
  --linker-l20-rs485-profile v10 \
  --linker-l20-command-hz 80 \
  --linker-l20-max-register-step 3'
```

### 11.3 最大软件速度

用户最后确认使用的参数：

```bash
cd /home/engram/AnyDexRetarget/example

sg dialout -c 'exec /home/engram/anaconda3/envs/anydex/bin/python teleop_real.py \
  --robot linker_l20 \
  --input pico4 \
  --pico4-mode relay \
  --hand left \
  --retarget-version v2 \
  --linker-l20-transport rs485 \
  --linker-l20-port /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0 \
  --linker-l20-baudrate 460800 \
  --linker-l20-slave-id 42 \
  --linker-l20-rs485-profile v10 \
  --linker-l20-command-hz 0 \
  --linker-l20-max-register-step 0'
```

其含义为：

$$
command\_hz=0\Rightarrow\text{不做软件发送频率限制}
$$

$$
max\_register\_step=0\Rightarrow\text{不做逐命令寄存器渐变}
$$

每个可用 Pico 新目标都会直接写入真机。该模式响应最快，但突然出现异常关键点时也会直接产生较大的命令变化，只应在机械手固定、周围清空、急停或断电手段可用时使用。

## 12. 暂停、手动掰动与安全

### 12.1 正常暂停

在运行 `teleop_real.py` 的终端按：

```text
Ctrl-C
```

程序会停止循环并关闭串口。当前 `open_on_exit` 默认为 false，因此停止时保持最后位置，不会自动张开。

若终端丢失，先只读查找精确 PID：

```bash
pgrep -af 'teleop_real.py.*linker_l20'
```

确认 PID 后发送可处理的中断信号：

```bash
kill -INT <PID>
```

不要使用模糊的 `pkill python`，避免停止 Pico daemon、仿真或其他 Python 任务。

### 12.2 需要手动掰机械手

当前已验证协议只覆盖位置读取和写入，没有确认 L20 的软件“失能/释放力矩”寄存器。停止 `teleop_real.py` 只代表不再发送新目标，不保证手端伺服立即释放保持力。

若关节仍有保持力：

1. 不要强行逆着电机扭矩掰动；
2. 使用官方支持的失能方法，或在确认安全后关闭手端电源；
3. 重新上电后先用安全低速模式接管；
4. 不要猜测速度、电流、使能或故障寄存器并直接写入真机。

## 13. 故障排查

### 13.1 USB-RS485 不出现

```bash
lsusb | rg '1a86:7523|CH340|CH341'
ls -l /dev/serial/by-id/
ls -l /dev/ttyUSB*
```

如果 CH340 在 `lsusb` 中存在但没有串口节点，检查内核日志和 `ch341-uart` 驱动。如果设备节点存在但权限不足，继续使用 `sg dialout -c` 或重新登录以刷新附加组。

### 13.2 Pico USB 网卡或 IP 不出现

```bash
ip -br addr | rg 'usb|enx|192\.168\.172\.'
ss -lunp | rg '63901'
pgrep -af 'pico4_daemon.py'
```

本次成功链路中，Pico 地址为 `192.168.172.29`，本机地址为 `192.168.172.89:63901`。若 USB 网络接口没有出现，应先在 Pico 端确认 USB 网络/串流模式，再重插 USB；仅启动 Python 程序不能凭空创建 USB 网卡。

### 13.3 串口存在但真机无响应

按以下顺序检查：

1. L20 手端电源和供电电压；
2. RS485 的 A/B 极性和公共 GND；
3. 稳定设备路径是否仍指向当前 CH340；
4. `460800 / 8N1 / ID 42` 是否保持不变；
5. 串口是否被另一个进程占用；
6. 启动只读 `FC04` 握手是否返回完整 65 字节和合法 CRC。

### 13.4 控制太慢

影响响应的两项软件参数为：

- `--linker-l20-command-hz`：最大命令频率；
- `--linker-l20-max-register-step`：每次寄存器最大变化量。

从低速到高速的建议顺序为：

```text
30 Hz / step 1
80 Hz / step 3
0 Hz limit / step 0
```

如果已使用最大软件速度但仍慢，瓶颈更可能来自 Pico 输入帧率、手端执行器速度、内部滤波或机械负载，不应继续通过超范围寄存器解决。

### 13.5 仿真能捏合但真机不能

依次记录同一帧的：

```text
Pico 21 点
变换后关键点
V2 qpos
目标寄存器
FC04 实际回读寄存器
实物指尖距离
```

这样可以把问题定位为输入、IK、映射、通信、执行器跟随或机械模型误差，而不是继续叠加不可迁移的常数补偿。

## 14. 换其他机械手时如何复用

推荐保持以下分层：

```text
通用层：AdaptiveOptimizerAnalyticalV2 + AdaptiveLPFilterV2
模型层：新手型 URDF/MJCF + mimic + joint limits
配置层：关键点旋转 + direct joint names + contact target
标定层：model qpos -> actuator command 单调分段点
驱动层：CAN/RS485/串口协议 + 寄存器布局
```

换手时的最小步骤：

1. 新增或确认新手型模型，验证 FK、关节轴和左右手镜像；
2. 为 `direct_control_v2.joint_names` 绑定每根手指的独立执行轴；
3. 测量张开、中位、闭合等安全标定点；
4. 用 `joint_command_mapping.py` 配置单调分段投影；
5. 设置各指 `target_distances_cm`，不能默认认为所有手都能达到零距离；
6. 编写或复用对应通信驱动，先只读握手，再原值回写，再做单通道小步动作；
7. 通过离线姿态、仿真 FK 和真机低速验证后再提高速度。

若新机械手没有独立手指执行器，算法不能创造硬件不存在的自由度。例如只有一个联动四指电机时，无法得到和人手完全一致的四指独立控制。

## 15. 已完成验证

本次开发阶段已获得以下验证结果：

| 验证项 | 结果 |
|---|---|
| V10 RS485 只读握手 | 通过 |
| 原位置值回写与回读 | 通过 |
| 食指 MCP 单通道小幅动作 | 通过 |
| standard 位置寄存器布局 | 通过 |
| 保留区 `15-24` 写零 | 通过 |
| 从真机当前位置渐变接管 | 通过 |
| 左手四指 MCP roll 最终方向 | 通过离线回归 |
| CLI 驱动参数传递 | 通过 |
| 最终小集合 | `9 passed, 4 subtests passed` |
| V2/映射/协议较大集合 | `40 passed` |
| L20 左右手 16 通道 FK | 通过 |
| 最大 FK 位置误差 | `8.870e-08 m` |
| 最大 FK 旋转误差 | `1.040e-06 rad` |
| 最大软件速度实测 | 控制约 `93-95 FPS`，Pico 约 `29-31 FPS` |

本次文档整理没有重新使能真机。以上真机运动结果来自 2026-08-27 已完成的实际联调记录。

## 16. 尚未完成的验证

以下项目不能声称已经完全解决：

- 无名指和小指与拇指在真机上的全姿态范围精确对指；
- 拇指在每个独立自由度上的动态跟随误差；
- 不同速度、负载和长期运行下的回差；
- 真机指尖几何测量与模型参数的系统标定；
- 软件失能/释放力矩寄存器；
- 换另一款手后，仅替换配置即可达到同等精度的真机验证。

后续继续优化时，应优先采集同步的输入、qpos、目标寄存器、回读寄存器和指尖实测位置，再修改手型标定；不应把新的 L20 固定补偿写进通用 V2 优化器。

## 17. 本轮结束状态

截至本文整理时：

```text
L20 真机 teleop_real.py：已停止
L20 串口：已关闭
MuJoCo/debug_skeleton 仿真：已停止
Pico daemon：仍运行，仅接收/转发数据
机械手退出动作：保持最后位置，没有自动回零
```

下次开始真机控制前，先检查机械手已固定、周围无障碍物、串口路径存在，再从“安全低速验证”启动。
