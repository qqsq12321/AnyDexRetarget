# Inspire RH56DFTP-2R / RH56E2 重定向算法 v2

## 1. 目标与结论

本版本面向 Pico 4 驱动 Inspire Robots `RH56DFTP-2R` 右手时的精细对指问题，作为独立 `v2` 路径提供，不替换现有 `AdaptiveOptimizerAnalytical` 和原配置。

现有 v1 在人手已经捏合时，主要分别拟合“腕部到各指尖”的目标，没有直接要求机器人拇指和目标指尖互相接近。因此两个独立目标都可以具有较小误差，但机器人指尖之间仍留有明显间隙。

v2 的核心改动有四项：

1. 非捏合状态下，把人手手指弯曲和拇指旋转/弯曲意图直接映射到 RH56 六个独立通道。
2. 拇指与任一手指形成明确双指捏合时，仅对该伙伴启用显式接触距离损失。
3. 两个候选手指同时接近且没有明显优势时，不启动接触优化，避免握拳或多指动作误触发。
4. 当前真机 v2 配置使用零延迟输出，拇指两个主动通道始终直通，然后重建 mimic 关节。

离线回放验证结果：

| 指标 | v1 | v2 | 改善 |
|---|---:|---:|---:|
| 拇指-食指明确捏合平均指尖距 | 33.29 mm | 4.70 mm | 85.88% |
| 拇指-中指明确捏合平均指尖距 | 28.64 mm | 12.41 mm | 56.67% |
| 拇指-无名指明确捏合平均指尖距 | 37.12 mm | 32.41 mm | 12.69% |
| 拇指-小指明确捏合平均指尖距 | 54.40 mm | 52.64 mm | 3.24% |
| v2 完整回放平均求解时间 | - | 9.64 ms | P95 15.53 ms |

首次真机验证中，用户确认拇指-食指对指有效，同时报告其他手指对指与拇指单独跟随不足。第二轮实现据此增加独立直接控制状态。

代表性孤立捏合帧上的 URDF 指尖中心距为：

| 捏合伙伴 | v1 | 新 v2 | 结论 |
|---|---:|---:|---|
| 食指 | 38.13 mm | 2.64 mm | 可接近几何闭合 |
| 中指 | 25.22 mm | 12.41 mm | 已到当前模型可达边界 |
| 无名指 | 36.38 mm | 32.45 mm | 受拇指旋转上限限制 |
| 小指 | 53.04 mm | 52.64 mm | 当前机构无法实现人手式真正对指 |

这些是指尖 frame 中心距，真实指腹软性接触仍需真机视觉或触觉数据标定。

## 2. RH56 结构确认

### 2.1 官方参数

因时官网当前将 `rh56dftp-series` 页面展示为 `RH56E2` 系列，主要结构参数为：

- 6 个主动自由度。
- 12 个运动关节。
- 6 个力传感器。
- 5-17 个触觉传感器。
- 重复定位精度 `±0.2 mm`。
- 指尖最大输出力 `30 N`。
- 拇指横向旋转范围大于 `85°`。

用户所述 `RH56DFTP-2R` 可能是同一机械平台的旧命名或具体配置型号；最终应以手背铭牌、固件版本和厂家参数表为准。

### 2.2 六通道顺序

官方 RH56 手册和 Unitree 开源控制器都给出以下顺序：

```text
[pinky, ring, middle, index, thumb_bend, thumb_rotation]
```

当前项目的真实输出映射为：

```python
_INSPIRE_CHANNEL_INDICES = [4, 6, 2, 0, 9, 8]
```

使用项目实际的 Pinocchio 模型打印得到：

```text
q[0] = index_proximal_joint
q[2] = middle_proximal_joint
q[4] = pinky_proximal_joint
q[6] = ring_proximal_joint
q[8] = thumb_proximal_yaw_joint
q[9] = thumb_proximal_pitch_joint
```

所以现有映射与六通道顺序一致，索引不是本次捏合误差的主因。

## 3. v1 数学模型及缺口

设人手关键点为：

$$
P = \{p_k \in \mathbb{R}^3\}_{k=0}^{20}
$$

机器人关节向量为：

$$
q \in \mathbb{R}^{12}
$$

RH56 URDF 包含 mimic 关系，优化器只搜索 6 个独立变量：

$$
q_{ind} =
\begin{bmatrix}
q_{index} & q_{middle} & q_{pinky} & q_{ring} & q_{thumb,yaw} & q_{thumb,pitch}
\end{bmatrix}^{\mathsf T}
$$

v1 根据人手拇指到其他指尖的距离计算捏合权重：

$$
\alpha_i =
\operatorname{clip}
\left(
\frac{d_{2,i}-d_i}{d_{2,i}-d_{1,i}+\varepsilon},
0,
\alpha_{max}
\right)
$$

其中：

$$
d_i = \left\|p_i^{tip}-p_{thumb}^{tip}\right\|_2
$$

v1 的每指目标主要由指尖位置、指尖方向和全手姿态组成：

$$
L_{v1}(q)
=
\sum_i
\left[
\alpha_i
\left(
w_{pos}L_{tip-pos,i}
+w_{dir}L_{tip-dir,i}
\right)
+
(1-\alpha_i)w_{full}L_{full,i}
\right]
+L_{reg}
$$

关节连续性正则项为：

$$
L_{reg}
=
\lambda_q\left\|q-q_{t-1}\right\|_2^2
$$

这里缺少直接的机器人指尖相对约束：

$$
\left\|x_i^{tip}(q)-x_{thumb}^{tip}(q)\right\|_2
$$

因此“人手捏合权重达到 1”只表示优化器更加重视两个独立指尖目标，并不保证机器人两个指尖真正闭合。

## 4. v2 接触距离算法

### 4.1 激活强度

v2 仍使用 v1 的捏合权重，但只在捏合足够明确时激活接触项。设原始归一化捏合强度为：

$$
\hat{\beta}_i
=
\operatorname{clip}
\left(
\frac{\alpha_i}{\alpha_{max}+\varepsilon},
0,
1
\right)
$$

配置激活阈值为 `beta_min = 0.8`，接触强度重新映射为：

$$
\beta_i
=
\operatorname{clip}
\left(
\frac{\hat{\beta}_i-\beta_{min}}
{1-\beta_{min}+\varepsilon},
0,
1
\right)
$$

这样在人手仅靠近、握拳或跟踪轻微误差时，不会过早把机器人指尖拉向接触位姿。

### 4.2 单一接触伙伴

RH56 只有 6 个主动自由度，不能同时精确复制人手多个接触关系。v2 允许四个非拇指手指成为候选，但只选择当前捏合权重最大的一个：

$$
i^*
=
\underset{i\in\{index,middle,ring,pinky\}}{\arg\max}\;\alpha_i
$$

设次大候选为 $i^{(2)}$，优势阈值为 $m=0.15$。如果两个候选都超过激活阈值，但优势不足，则不启动接触模式：

$$
\alpha_{i^*}-\alpha_{i^{(2)}} < m
\Longrightarrow
i^*=\varnothing
$$

这使 `thumb_yaw`、`thumb_bend` 和目标手指弯曲自由度只在“拇指 + 唯一目标指”时集中协同。

### 4.3 可达接触目标

设人手指尖距离为：

$$
d_i^h
=
100\left\|p_i^{tip}-p_{thumb}^{tip}\right\|_2
\quad [\mathrm{cm}]
$$

v2 对四个候选手指都下发接近闭合的期望距离：

$$
d_{desired,i}
=
0.20\ \mathrm{cm}
$$

这是期望目标，不是机构可达承诺。优化器受关节上下限约束，不可达的中指、无名指和小指会停在各自的几何边界。

正常手势距离先按原 `pinch_scaling` 缩放，并保证不小于机器人可达目标：

$$
d_{normal,i}
=
\max\left(s_p d_i^h,d_{desired,i}\right)
$$

最终参考距离随捏合强度连续过渡：

$$
d_{ref,i}
=
(1-\beta_i)d_{normal,i}
+\beta_i d_{desired,i}
$$

### 4.4 接触损失

机器人拇指到目标指尖的向量为：

$$
\Delta x_i(q)
=
x_i^{tip}(q)-x_{thumb}^{tip}(q)
$$

距离残差为：

$$
r_i(q)
=
\left\|\Delta x_i(q)\right\|_2-d_{ref,i}
$$

v2 增加 Huber 接触损失：

$$
L_{contact,i}(q)
=
w_c\beta_i\rho_{\delta_c}\left(r_i(q)\right)
$$

其中 Huber 函数为：

$$
\rho_{\delta}(r)
=
\begin{cases}
\frac{1}{2}r^2, & |r|\le\delta \\
\delta\left(|r|-\frac{1}{2}\delta\right), & |r|>\delta
\end{cases}
$$

总目标为：

$$
L_{v2}(q)
=
L_{v1}(q)
+L_{contact,i^*}(q)
$$

### 4.5 解析梯度

接触距离对关节的导数为：

$$
\frac{\partial r_i}{\partial q}
=
\frac{\Delta x_i^{\mathsf T}}
{\left\|\Delta x_i\right\|_2+\varepsilon}
\left(J_i-J_{thumb}\right)
$$

因此：

$$
\nabla_q L_{contact,i}
=
w_c\beta_i
\rho_{\delta_c}'(r_i)
\frac{\Delta x_i^{\mathsf T}}
{\left\|\Delta x_i\right\|_2+\varepsilon}
\left(J_i-J_{thumb}\right)
$$

实现继续使用 Pinocchio 批量 FK/Jacobian 和 NLopt `LD_SLSQP`。测试用中心有限差分校验了解析接触梯度。

### 4.6 独立手指直接控制

为了避免非捏合时的全手 IK 串扰，v2 从每个人手手指的三段方向计算总弯曲角：

$$
\phi_f
=
\arccos\left(\hat{s}_{f,1}^{\mathsf T}\hat{s}_{f,2}\right)
+
\arccos\left(\hat{s}_{f,2}^{\mathsf T}\hat{s}_{f,3}\right)
$$

弯曲角线性映射到该指的 RH56 独立关节范围：

$$
u_f
=
\operatorname{clip}
\left(
\frac{\phi_f-\phi_{open}}
{\phi_{closed}-\phi_{open}+\varepsilon},
0,1
\right)
$$

$$
q_f^{direct}
=
q_f^{min}
+u_f\left(q_f^{max}-q_f^{min}\right)
$$

拇指旋转使用掌宽归一化的横向位置。设食指指根到小指指根的单位向量为 $\hat{e}_{lat}$，掌宽为 $w_p$，则：

$$
r_{thumb}
=
\frac{
\left(p_{thumb}^{tip}-p_{wrist}\right)^{\mathsf T}
\hat{e}_{lat}
}{w_p+\varepsilon}
$$

$$
q_{thumb,yaw}^{direct}
=
q_{yaw}^{min}
+
\operatorname{clip}
\left(
\frac{r_{thumb}-r_{open}}
{r_{opposed}-r_{open}+\varepsilon},0,1
\right)
\left(q_{yaw}^{max}-q_{yaw}^{min}\right)
$$

拇指弯曲通道使用与其他手指相同的两段夹角和。非捏合时，六个独立通道完全采用 $q^{direct}$；明确捏合时，仅拇指和接触伙伴保留 IK/接触优化结果，其他手指仍采用独立直接目标。

## 5. v2 自适应滤波

固定 EMA 为：

$$
y_t
=
y_{t-1}+\alpha(x_t-y_{t-1})
$$

它会对所有动作引入相位延迟。v2 对每个关节按其行程定义直通阈值：

$$
\tau_j
=
r_b\left(q_j^{max}-q_j^{min}\right)
$$

当前配置 `r_b = 0.008`，约等于真机 0-1000 量程中的 8 counts。但针对用户报告的拇指跟随延迟，真机 v2 配置将 $\alpha_f$ 设为 `1.0`，因此当前等价为零延迟输出；阈值机制仍保留，便于以后根据真机抖动数据恢复小信号平滑。

滤波规则为：

$$
y_{t,j}
=
\begin{cases}
x_{t,j}, & |x_{t,j}-y_{t-1,j}|\ge\tau_j \\
y_{t-1,j}+\alpha_f(x_{t,j}-y_{t-1,j}), & \text{otherwise}
\end{cases}
$$

拇指旋转和拇指弯曲两个通道还显式列入 `passthrough_joints`，避免后续调整其他手指滤波时再引入拇指相位延迟。

RH56 的 12 维 URDF 中只有 6 个独立变量。实现只对这 6 个主动变量滤波，再通过 mimic 关系重建其余关节：

$$
q_{mimic,j}
=
k_j q_{source,j}+b_j
$$

因此 v2 仿真不会因为逐关节滤波而暂时破坏机械耦合关系。

## 6. 文件与启用方式

新增文件：

```text
anydexretarget/optimizer/analytical_optimizer_v2.py
anydexretarget/optimizer/filter_v2.py
example/config/adaptive/pico4/pico4_inspire_hand_v2.yaml
example/test/test_adaptive_optimizer_v2.py
INSPIRE_RH56_RETARGETING_V2.zh.md
```

v2 真机命令示例：

```bash
cd /home/engram/AnyDexRetarget/example
conda run -n anydex python teleop_real.py \
  --robot inspire \
  --input pico4 \
  --hand right \
  --pico4-mode relay \
  --config config/adaptive/pico4/pico4_inspire_hand_v2.yaml \
  --inspire-port /dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A10PH40A-if00-port0
```

在串口权限和安全检查完成前，不应直接运行真机命令。可先做离线回放：

```bash
cd /home/engram/AnyDexRetarget/example
conda run -n anydex python teleop_sim.py \
  --robot inspire \
  --play data/avp1.pkl \
  --config config/adaptive/pico4/pico4_inspire_hand_v2.yaml
```

回归测试：

```bash
cd /home/engram/AnyDexRetarget
conda run -n anydex python -m pytest -q \
  example/test/test_adaptive_optimizer_v2.py
```

## 7. 真机验证与标定顺序

### 7.1 只读确认

1. 将当前用户加入 `dialout`，重新登录后确认串口可访问。
2. 只读取手 ID、`ANGLE_ACT`、`STATUS` 和 `ERROR`。
3. 确认真机六通道顺序、方向和当前位置，没有错误或过流状态。

### 7.2 小行程检查

1. 保持手周围无物体、可立即断电。
2. 每次只改变一个通道约 10-20 counts。
3. 确认第 6 通道确实是拇指横向旋转，第 5 通道是拇指弯曲。
4. 验证控制值方向与驱动映射一致。

### 7.3 捏合标定

录制至少三类 Pico 数据：

- 完全张手 5 秒。
- 拇指-食指缓慢接近、接触、离开，各重复 10 次。
- 拇指-中指缓慢接近、接触、离开，各重复 10 次。
- 拇指-无名指和拇指-小指各重复 10 次，记录最小可达间隙，不强行宣称能实现真正指腹接触。
- 分别单独弯曲食指、中指、无名指、小指，并单独测试拇指旋转和弯曲。

每帧保存：

$$
\left{
d_h,
d_r,
q,
u_{0:5},
ANGLE\_ACT_{0:5},
FORCE\_ACT_{0:5}
\right}
$$

其中 `d_h` 是人手指尖距离，`d_r` 是 URDF 机器人指尖距离，`u` 是六路命令。若有触觉/力反馈，应使用首次稳定接触帧重新拟合 `d_reach`，而不是仅凭视觉估计。

## 8. 已知限制

- RH56 只有 6 个主动自由度，无法完整复制人手 20+ 自由度的精细姿态。
- 当前 URDF 指尖 frame 与真实指腹接触面可能存在毫米级偏差。
- 中指、无名指和小指在当前模型中的最小中心距约为 `12.38 mm`、`32.38 mm`、`52.61 mm`；真机指腹软性可缩小视觉间隙，但不能突破拇指旋转和指根布局的机械约束。
- v2 已允许四个非拇指手指进入接触模式，但通过 `dominance_margin` 拒绝多指歧义帧。
- 真机已实测稳定运行在约 `62 FPS` 控制和 `111-129 FPS` Pico 输入；串口六路速度寄存器均为最大值 `1000`，拇指滞后不是速度寄存器过低造成的。
- 力控寄存器可以在接触任务中提供保护，但未经真机安全标定，本版本不自动修改力控阈值。

## 9. 参考资料

1. Inspire Robots RH56E2 产品页：<https://www.inspire-robots.com/dexterous%20hands/rh56dftp-series/>
2. 《RH56 系列用户手册》，文档编号 `PRJ-02-TS-U-001`，2023-11-17 镜像：<https://github.com/Hao-Starrr/ins-dex-retarget/blob/main/assets/%E5%9B%A0%E6%97%B6%E6%89%8B%E6%96%87%E6%A1%A3.pdf>
3. Unitree Inspire 控制服务：<https://github.com/unitreerobotics/dfx_inspire_service>
4. Inspire Hand SDK，含 `SPEED_SET=1522` 和 `DEFAULT_SPEED_SET=1032` 定义：<https://github.com/Director-of-G/inspire_hand_driver>
5. dex-retargeting，DexPilot 实现，检索提交 `3f56141bc8bd2760d5e452e382937269554ebb21`：<https://github.com/dexsuite/dex-retargeting>
6. Handa et al., “DexPilot: Vision Based Teleoperation of Dexterous Robotic Hand-Arm System”：<https://arxiv.org/abs/1910.03135>
7. Qin et al., “AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System”, RSS 2023：<https://yzqin.github.io/anyteleop/>
8. Inspire 6-DOF 线性映射参考，检索提交 `086a47dbef55f18115538fadba613588e2acd759`：<https://github.com/Hao-Starrr/ins-dex-retarget>
9. Quest/Inspire 低延迟调优经验，检索提交 `c841cb66f1d83057bb51d6c31ec55ee92dc59bbc`：<https://github.com/GeneralTrajectory/dex-teleop>
