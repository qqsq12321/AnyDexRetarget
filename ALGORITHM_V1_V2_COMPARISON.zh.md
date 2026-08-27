# 原算法（v1）与 v2 算法异同点

## 1. 文档范围

本文对比 AnyDexRetarget 中以下两个实现：

- 原算法（下文简称 **v1**）：`AdaptiveOptimizerAnalytical`
- v2 算法：`AdaptiveOptimizerAnalyticalV2`

主要对应源码与配置：

- `anydexretarget/optimizer/analytical_optimizer.py`
- `anydexretarget/optimizer/analytical_optimizer_v2.py`
- `anydexretarget/optimizer/filter_v2.py`
- `anydexretarget/optimizer/base_optimizer.py`
- `anydexretarget/retarget.py`
- `example/config/adaptive/pico4/pico4_inspire_hand.yaml`
- `example/config/adaptive/pico4/pico4_inspire_hand_v2.yaml`

本文中的 v2 特性主要面向 Inspire RH56DFTP-2R / RH56E2 六主动通道手型。v2 没有删除或替换 v1，而是通过继承复用 v1 的 IK、目标函数和解析梯度，再增加独立通道控制、明确捏合伙伴选择、指尖接触约束和低延迟滤波机制。

## 2. 结论概览

| 对比项 | v1 原算法 | v2 算法 |
|---|---|---|
| 基本方法 | 每帧运行有界非线性优化 | 非捏合时可直接控制；明确捏合时运行 v1 IK 并增加接触项 |
| 类关系 | 继承 `BaseOptimizer` | 继承 `AdaptiveOptimizerAnalytical` |
| 全手姿态 | 通过 PIP、DIP、TIP 三组腕部相对向量匹配 | 捏合优化阶段继续复用同一目标 |
| 捏合表示 | 用连续权重在全手姿态与指尖目标之间混合 | 复用 v1 权重，并从候选中选出唯一捏合伙伴 |
| 接触闭合 | 没有直接约束两机器人指尖间距 | 增加拇指与伙伴指尖的显式距离损失 |
| 普通单指运动 | 各手指仍处于同一个 IK 问题中 | 可将每根手指直接映射到独立主动关节 |
| 模糊/多指捏合 | 多根手指可同时具有较高捏合权重 | 不满足唯一性条件时拒绝进入接触状态 |
| 非捏合计算量 | 始终运行 SLSQP | `direct_only: true` 时跳过 SLSQP |
| Mimic joint | 优化独立关节后重建 | 相同；直接控制和滤波后也重新重建 |
| 输出滤波 | 全量关节统一 EMA 低通 | 独立关节按变化幅度旁路，再重建 mimic joint |
| 主要优势 | 通用、连续、保持整体手型 | 独立单指响应更直接，双指接触更可靠，非捏合延迟更低 |
| 主要代价 | 接触距离不受显式保证，耦合机械手上可能难以闭合 | 参数和状态逻辑更多，直接映射依赖具体机械手通道语义 |

一句话概括：

> v1 是“全程通过统一 IK 在姿态与捏合目标之间连续权衡”；v2 是“非捏合时直接控制，明确捏合时在 v1 IK 上增加显式接触距离约束”。

## 3. 共同输入、变量与单位

两种算法接收相同的 21 点手部关键点：

$$
\mathbf P=
\left\{
\mathbf p_0,\mathbf p_1,\ldots,\mathbf p_{20}
\right\},
\qquad
\mathbf p_k\in\mathbb R^3
$$

其中关键点输入单位为 m。优化器内部的位置与距离统一转换为 cm：

$$
\mathbf p_k^{cm}=100\mathbf p_k
$$

设：

| 符号 | 含义 |
|---|---|
| $\mathbf q$ | 完整机器人关节向量 |
| $\mathbf q_a$ | 独立主动关节向量 |
| $\mathbf q_{t-1}$ | 上一帧关节向量 |
| $\mathbf x_i^{tip}(\mathbf q)$ | 机器人第 $i$ 根手指的指尖位置 |
| $\mathbf x_i^{pip}(\mathbf q)$ | 机器人第 $i$ 根手指的 PIP 对应点 |
| $\mathbf x_i^{dip}(\mathbf q)$ | 机器人第 $i$ 根手指的 DIP 对应点 |
| $J_i(\mathbf q)$ | 对应位置关于 $\mathbf q$ 的 Jacobian |
| $\alpha_i$ | v1 的第 $i$ 根手指捏合权重 |
| $\beta$ | v2 接触状态的归一化激活强度 |
| $\varepsilon$ | 防止除零的小常数，代码中约为 $10^{-8}$ |

## 4. 两种算法共同复用的基础

### 4.1 Huber 鲁棒损失

v1 的位置、方向和全手姿态误差，以及 v2 新增的接触距离误差，均使用 Huber 损失：

$$
\rho_\delta(r)=
\begin{cases}
\dfrac{1}{2}r^2, & |r|\le \delta,\\[6pt]
\delta\left(|r|-\dfrac{1}{2}\delta\right), & |r|>\delta.
\end{cases}
$$

其导数为：

$$
\rho_\delta'(r)=
\begin{cases}
r, & |r|\le \delta,\\[4pt]
\delta\operatorname{sign}(r), & |r|>\delta.
\end{cases}
$$

Huber 损失在误差较小时保持二次函数的平滑性，在误差较大时转为线性增长，降低异常关键点对优化结果的影响。

### 4.2 分段骨架缩放

两种算法都保留 v1 的 `segment_scaling` 目标生成方式。对第 $i$ 根手指，设四段缩放系数为：

$$
\mathbf s_i=
\begin{bmatrix}
s_{i,mcp} & s_{i,pip} & s_{i,dip} & s_{i,tip}
\end{bmatrix}^{\mathsf T}
$$

从腕部开始逐段累积：

$$
\mathbf t_{i,mcp}
=
s_{i,mcp}
\left(\mathbf p_{i,mcp}-\mathbf p_w\right)
$$

$$
\mathbf t_{i,pip}
=
\mathbf t_{i,mcp}
+
s_{i,pip}
\left(\mathbf p_{i,pip}-\mathbf p_{i,mcp}\right)
$$

$$
\mathbf t_{i,dip}
=
\mathbf t_{i,pip}
+
s_{i,dip}
\left(\mathbf p_{i,dip}-\mathbf p_{i,pip}\right)
$$

$$
\mathbf t_{i,tip}
=
\mathbf t_{i,dip}
+
s_{i,tip}
\left(\mathbf p_{i,tip}-\mathbf p_{i,dip}\right)
$$

因此目标不仅缩放总长度，还能分别校正掌部跨度和每段指骨长度。

### 4.3 有界 SLSQP 与解析梯度

需要运行 IK 时，两种算法使用相同的 NLopt `LD_SLSQP` 有界优化器：

$$
\mathbf q_a^*
=
\underset{\mathbf q_a}{\arg\min}\;L(\mathbf q_a)
$$

满足关节上下界：

$$
\mathbf q_{a,min}
\le
\mathbf q_a
\le
\mathbf q_{a,max}
$$

代码向 SLSQP 提供手写解析梯度，并以上一帧结果作为下一帧初值，从而减少迭代量并保持时序连续性。

### 4.4 Mimic joint 约束

两种算法只优化独立关节。对任一 mimic joint：

$$
q_m=a_mq_s+b_m
$$

其中 $q_s$ 为源主动关节，$a_m$ 和 $b_m$ 分别来自 URDF 的 `multiplier` 与 `offset`。

梯度通过链式法则映射回主动关节：

$$
\frac{\partial L}{\partial q_s}
\leftarrow
\frac{\partial L}{\partial q_s}
+
a_m\frac{\partial L}{\partial q_m}
$$

## 5. v1 原算法

### 5.1 捏合检测权重

对每根非拇指手指 $i$，计算人手拇指指尖与该指尖的距离：

$$
d_i^h
=
100
\left\|
\mathbf p_{i,tip}-\mathbf p_{thumb,tip}
\right\|_2
$$

根据两级阈值 $d_{1,i}$、$d_{2,i}$ 计算捏合权重：

$$
\alpha_i
=
\operatorname{clip}
\left(
\frac{d_{2,i}-d_i^h}
{d_{2,i}-d_{1,i}+\varepsilon},
0,
\alpha_{max}
\right)
$$

拇指权重取全部非拇指手指中的最大值：

$$
\alpha_{thumb}
=
\max_{i\ne thumb}\alpha_i
$$

这是一种连续权重，而不是硬开关：距离越近，目标越偏向指尖捏合；距离越远，目标越偏向全手姿态。

### 5.2 捏合时的指尖目标修正

v1 找出当前权重最大的非拇指伙伴：

$$
k
=
\underset{i\ne thumb}{\arg\max}\;\alpha_i
$$

将其权重归一化为：

$$
\gamma
=
\min
\left(
\frac{\alpha_k}{\alpha_{max}+\varepsilon},
1
\right)
$$

腕部到指尖的统一缩放目标为：

$$
\mathbf t_{i,tip}^{pinch}
=
s_{pinch}
\left(
\mathbf p_{i,tip}-\mathbf p_w
\right)
$$

仅对拇指和伙伴手指 $i\in\{thumb,k\}$ 插值：

$$
\widetilde{\mathbf t}_{i,tip}
=
(1-\gamma)\mathbf t_{i,tip}
+
\gamma\mathbf t_{i,tip}^{pinch}
$$

该修正改变的是“腕部到各自指尖”的目标向量，并没有直接约束机器人拇指与伙伴指尖之间的距离。

### 5.3 v1 指尖位置损失

机器人第 $i$ 根手指的指尖向量为：

$$
\mathbf v_i^{r,pos}(\mathbf q)
=
\mathbf x_i^{tip}(\mathbf q)
-
\mathbf x_i^{origin}(\mathbf q)
$$

位置残差为：

$$
r_i^{pos}
=
\left\|
\mathbf v_i^{r,pos}(\mathbf q)
-
\widetilde{\mathbf t}_{i,tip}
\right\|_2
$$

位置损失为：

$$
L_i^{pos}
=
w_{pos}\rho_{\delta_{pos}}\left(r_i^{pos}\right)
$$

### 5.4 v1 指尖方向损失

机器人指尖方向使用末端两点归一化：

$$
\widehat{\mathbf u}_i^r(\mathbf q)
=
\frac{
\mathbf x_i^{tip}(\mathbf q)-\mathbf x_i^{link4}(\mathbf q)
}{
\left\|
\mathbf x_i^{tip}(\mathbf q)-\mathbf x_i^{link4}(\mathbf q)
\right\|_2+\varepsilon
}
$$

人手目标方向为：

$$
\widehat{\mathbf u}_i^h
=
\frac{
\mathbf p_{i,tip}-\mathbf p_{i,dip}
}{
\left\|
\mathbf p_{i,tip}-\mathbf p_{i,dip}
\right\|_2+\varepsilon
}
$$

方向损失为：

$$
L_i^{dir}
=
w_{dir}
\rho_{\delta_{dir}}
\left(
\left\|
\widehat{\mathbf u}_i^r-\widehat{\mathbf u}_i^h
\right\|_2
\right)
$$

### 5.5 v1 全手姿态损失

对 PIP、DIP、TIP 三个目标点分别计算腕部相对向量误差。设：

$$
r_{i,j}^{full}
=
\left\|
\left(
\mathbf x_{i,j}(\mathbf q)-\mathbf x_w(\mathbf q)
\right)
-
\mathbf t_{i,j}
\right\|_2,
\qquad
j\in\{pip,dip,tip\}
$$

则单指全手姿态损失为：

$$
L_i^{full}
=
\frac{w_{full}}{3}
\sum_{j\in\{pip,dip,tip\}}
\rho_{\delta_{pos}}
\left(r_{i,j}^{full}\right)
$$

### 5.6 v1 总目标函数

v1 对每根手指在指尖目标和全手姿态目标之间进行混合：

$$
L_{v1}(\mathbf q)
=
\sum_i
\left[
\alpha_i
\left(
L_i^{pos}+L_i^{dir}
\right)
+
(1-\alpha_i)L_i^{full}
\right]
+
L_{reg}
$$

帧间正则项为：

$$
L_{reg}
=
\lambda
\left\|
\mathbf q-\mathbf q_{t-1}
\right\|_2^2
$$

其核心特点是：所有手指始终由一个统一的 IK 目标共同求解，只是每根手指的损失权重不同。

## 6. v2 算法新增内容

### 6.1 v2 的两种执行路径

v2 每帧先判断是否存在唯一且明确的捏合伙伴。状态可写为：

$$
S_t=
\begin{cases}
DIRECT, & \text{不存在明确捏合伙伴},\\[4pt]
PINCH(k), & \text{拇指与第 }k\text{ 根手指明确捏合}.
\end{cases}
$$

当 `direct_control_v2.direct_only: true` 时（当前实现默认值为 `true`，Inspire v2 YAML 未显式覆盖）：

$$
\mathbf q_t=
\begin{cases}
\mathbf q_t^{direct}, & S_t=DIRECT,\\[4pt]
\operatorname{Blend}
\left(
\mathbf q_t^{IK+contact},
\mathbf q_t^{direct}
\right), & S_t=PINCH(k).
\end{cases}
$$

因此非捏合帧可以完全跳过 SLSQP；只有进入明确捏合状态时才调用继承自 v1 的优化过程。

### 6.2 非拇指手指直接控制

对食指、中指、无名指和小指，v2 使用三段人手向量：

$$
\mathbf v_1=\mathbf p_{pip}-\mathbf p_{mcp}
$$

$$
\mathbf v_2=\mathbf p_{dip}-\mathbf p_{pip}
$$

$$
\mathbf v_3=\mathbf p_{tip}-\mathbf p_{dip}
$$

两段向量的夹角定义为：

$$
\theta(\mathbf a,\mathbf b)
=
\arccos
\left[
\operatorname{clip}
\left(
\frac{\mathbf a^{\mathsf T}\mathbf b}
{(\|\mathbf a\|_2+\varepsilon)(\|\mathbf b\|_2+\varepsilon)},
-1,
1
\right)
\right]
$$

单指总弯曲量为：

$$
c_i
=
\theta(\mathbf v_1,\mathbf v_2)
+
\theta(\mathbf v_2,\mathbf v_3)
$$

将弯曲量映射到 $[0,1]$：

$$
r_i
=
\operatorname{clip}
\left(
\frac{c_i-c_{open}}
{c_{closed}-c_{open}+\varepsilon},
0,
1
\right)
$$

再映射到对应主动关节范围：

$$
q_i^{direct}
=
q_{i,min}
+
r_i
\left(q_{i,max}-q_{i,min}\right)
$$

每根手指独立计算 $c_i$，因此单独弯曲一根手指不会主动改变其他三根手指的直接目标。

### 6.3 拇指直接控制

v2 将拇指拆为旋转和弯曲两个主动通道。

定义掌宽方向：

$$
\mathbf l
=
\mathbf p_{pinky,mcp}-\mathbf p_{index,mcp}
$$

$$
w_p=\|\mathbf l\|_2,
\qquad
\widehat{\mathbf e}_{lat}
=
\frac{\mathbf l}{w_p+\varepsilon}
$$

拇指横向位置比例为：

$$
h_{thumb}
=
\frac{
\left(\mathbf p_{thumb,tip}-\mathbf p_w\right)^{\mathsf T}
\widehat{\mathbf e}_{lat}
}{w_p+\varepsilon}
$$

归一化并映射到拇指旋转关节：

$$
r_{yaw}
=
\operatorname{clip}
\left(
\frac{h_{thumb}-h_{open}}
{h_{opposed}-h_{open}+\varepsilon},
0,
1
\right)
$$

$$
q_{thumb,yaw}^{direct}
=
q_{yaw,min}
+
r_{yaw}
\left(q_{yaw,max}-q_{yaw,min}\right)
$$

拇指弯曲量与普通手指相同，由拇指 MCP、PIP、DIP、TIP 三段的两个夹角之和得到，再映射到拇指弯曲关节范围。

### 6.4 唯一捏合伙伴选择

设启用的非拇指候选集合为 $\mathcal F$。先按 $\alpha_i$ 从大到小排列：

$$
k
=
\underset{i\in\mathcal F}{\arg\max}\;\alpha_i
$$

次大候选记为 $k_2$。如果次大候选也超过激活门槛，且两者差值小于优势间隔：

$$
\alpha_{k_2}\ge \beta_{min}
\quad\land\quad
\alpha_k-\alpha_{k_2}<m_{dom}
$$

则 v2 认为当前捏合伙伴不明确，不进入接触状态。

对主候选先计算归一化强度：

$$
\beta_{raw}
=
\operatorname{clip}
\left(
\frac{\alpha_k}{\alpha_{max}+\varepsilon},
0,
1
\right)
$$

只有满足：

$$
\beta_{raw}\ge\beta_{min}
$$

才激活接触。激活后的平滑强度为：

$$
\beta
=
\operatorname{clip}
\left(
\frac{\beta_{raw}-\beta_{min}}
{1-\beta_{min}+\varepsilon},
0,
1
\right)
$$

该逻辑使 v2 只处理一个明确的双指捏合，避免两个候选接近时接触目标来回切换。

### 6.5 v2 接触目标距离

设人手拇指与伙伴手指的距离为：

$$
d_k^h
=
100
\left\|
\mathbf p_{k,tip}-\mathbf p_{thumb,tip}
\right\|_2
$$

先结合 `pinch_scaling`，并用机械可达目标距离 $d_k^{reach}$ 设置下限：

$$
d_k^{scaled}
=
\max
\left(
s_{pinch}d_k^h,
d_k^{reach}
\right)
$$

最终接触目标为：

$$
d_k^{target}
=
(1-\beta)d_k^{scaled}
+
\beta d_k^{reach}
$$

随着 $\beta$ 从 0 增大到 1，目标距离从缩放后的人手距离逐渐收紧到机械手标定的可达距离。

### 6.6 v2 显式指尖接触损失

机器人拇指与伙伴指尖的实际距离为：

$$
d_k^r(\mathbf q)
=
\left\|
\mathbf x_k^{tip}(\mathbf q)
-
\mathbf x_{thumb}^{tip}(\mathbf q)
\right\|_2
$$

接触残差为：

$$
r_k^{contact}
=
d_k^r(\mathbf q)-d_k^{target}
$$

v2 新增损失：

$$
L_{contact}
=
w_{contact}\beta
\rho_{\delta_{contact}}
\left(r_k^{contact}\right)
$$

设：

$$
\mathbf z_k
=
\mathbf x_k^{tip}-\mathbf x_{thumb}^{tip}
$$

$$
J_k^{rel}
=
J_k^{tip}-J_{thumb}^{tip}
$$

则接触距离对关节的梯度为：

$$
\frac{\partial d_k^r}{\partial\mathbf q}
=
\frac{\mathbf z_k^{\mathsf T}}
{\|\mathbf z_k\|_2+\varepsilon}
J_k^{rel}
$$

接触损失梯度为：

$$
\nabla_{\mathbf q}L_{contact}
=
w_{contact}\beta
\rho_{\delta_{contact}}'
\left(r_k^{contact}\right)
\frac{\partial d_k^r}{\partial\mathbf q}
$$

### 6.7 v2 优化目标

进入明确捏合状态时，v2 的优化目标为：

$$
L_{v2}(\mathbf q)
=
L_{v1}(\mathbf q)
+
L_{contact}(\mathbf q)
$$

因此 v2 没有改变 v1 的主要 IK 损失，而是通过 `_extra_loss_and_grad_v2()` 扩展钩子增加接触项。

### 6.8 IK 结果与直接控制结果混合

对每个主动通道 $j$，混合形式为：

$$
q_j^{out}
=
(1-b_j)q_j^{IK+contact}
+
b_jq_j^{direct}
$$

按当前 Inspire v2 配置及代码默认值：

- 非捏合：`direct_only` 使用默认值 `true`，直接返回 $\mathbf q^{direct}$。
- 明确捏合：伙伴手指使用 `pinch_finger_blend: 0.0`，保留 IK 与接触结果。
- 明确捏合：拇指使用 `pinch_thumb_blend: 0.0`，保留 IK 与接触结果。
- 其他非伙伴手指使用 `finger_blend: 1.0`，保持各自直接控制。

也就是说，明确捏合时只有“拇指 + 当前伙伴手指”由接触优化主导，其他手指仍独立跟随人手。

## 7. 输出滤波差异

### 7.1 v1 统一低通滤波

v1 使用普通指数移动平均：

$$
\mathbf y_t
=
\mathbf y_{t-1}
+
\eta
\left(
\mathbf x_t-\mathbf y_{t-1}
\right)
$$

其中 $\eta$ 为 `lp_alpha`。$\eta$ 越小，平滑越强，但响应延迟越大。

### 7.2 v2 按关节变化量旁路

v2 先对每个独立关节计算旁路阈值：

$$
\tau_j
=
r_{bypass}
\left(q_{j,max}-q_{j,min}\right)
$$

设当前变化量为：

$$
\Delta q_j
=
x_{t,j}-y_{t-1,j}
$$

则：

$$
y_{t,j}
=
\begin{cases}
x_{t,j}, & |\Delta q_j|\ge\tau_j
\text{，或该关节配置为 passthrough},\\[4pt]
y_{t-1,j}+\eta\Delta q_j, & |\Delta q_j|<\tau_j.
\end{cases}
$$

滤波只作用于独立关节，随后再按 URDF 关系重建 mimic joint，避免滤波破坏机械耦合关系。

需要注意：当前 `pico4_inspire_hand_v2.yaml` 中 `lp_filter_v2.alpha: 1.0`。此时即使进入 EMA 分支，也有：

$$
y_{t,j}
=
y_{t-1,j}+1\cdot\Delta q_j
=
x_{t,j}
$$

所以当前配置的实际效果是全部关节无平滑延迟地通过；若希望抑制微小抖动，应将 `alpha` 设置为小于 1 的值。

## 8. 关键相同点

1. 输入均为经过相同坐标变换和可选 `mediapipe_rotation` 后的 21 点手部骨架。
2. v2 继承 v1，因此捏合优化阶段共用分段骨架目标、指尖位置、指尖方向、全手姿态和帧间正则项。
3. 两者均使用 cm 作为优化内部的位置单位。
4. 两者均使用 Huber 损失降低异常关键点影响。
5. 两者在运行 IK 时都使用相同的 FK、Jacobian、解析梯度和有界 SLSQP。
6. 两者都只优化独立关节，并根据 URDF 重建 mimic joint。
7. 两者都使用上一帧结果作为优化初值或时序参考。

## 9. 关键不同点

### 9.1 控制思想

- v1：始终把整只手放入统一优化问题。
- v2：把普通跟随和接触闭合拆成两类控制问题。

### 9.2 捏合定义

- v1：每根手指都有连续 $\alpha_i$，允许多根手指同时具有较高权重。
- v2：在 $\alpha_i$ 基础上进一步选择一个唯一伙伴；不明确时拒绝接触优化。

### 9.3 接触保证

- v1：匹配各自的腕部到指尖向量，接触是间接结果。
- v2：直接最小化机器人两指尖距离与目标距离的差，接触是显式目标。

### 9.4 单指独立性

- v1：单指动作仍可能通过共享优化、机械耦合和全手目标影响其他关节。
- v2：非捏合手指可直接映射到各自主动通道，目标层面彼此独立。

### 9.5 计算路径

- v1：每帧都运行 SLSQP。
- v2：默认配置下，非捏合帧跳过 SLSQP；明确捏合帧才运行优化。

### 9.6 适用范围

- v1：对不同机器人更通用，主要依赖链路点和标定比例。
- v2：直接控制部分依赖具体主动关节名称和通道含义，更偏向 RH56 一类耦合手。

## 10. 当前 Inspire 配置参数对照

| 参数 | v1 | v2 | 作用 |
|---|---:|---:|---|
| `optimizer.type` | `AdaptiveOptimizerAnalytical` | `AdaptiveOptimizerAnalyticalV2` | 选择算法 |
| `huber_delta` | 2.0 | 2.0 | 位置与全手损失 Huber 阈值 |
| `huber_delta_dir` | 0.5 | 0.5 | 方向损失 Huber 阈值 |
| `norm_delta` | 0.04 | 0.04 | 帧间正则权重 |
| `w_pos` | 5.0 | 5.0 | 指尖位置权重 |
| `w_dir` | 1.0 | 1.0 | 指尖方向权重 |
| `w_full_hand` | 1.0 | 1.0 | 全手姿态权重 |
| `pinch_scaling` | 1.3224 | 1.3224 | 捏合目标尺度 |
| `alpha` | 1.0 | 1.0 | 最大捏合权重 |
| `lp_alpha` | 0.4 | - | v1 统一低通系数 |
| `contact_v2.weight` | - | 160.0 | v2 接触损失权重 |
| `contact_v2.huber_delta` | - | 0.5 | v2 接触损失阈值 |
| `contact_v2.min_beta` | - | 0.8 | 接触激活门槛 |
| `contact_v2.dominance_margin` | - | 0.15 | 唯一伙伴优势间隔 |
| `direct_control_v2.finger_blend` | - | 1.0 | 非伙伴手指直接控制比例 |
| `direct_control_v2.pinch_finger_blend` | - | 0.0 | 捏合伙伴直接控制比例 |
| `direct_control_v2.pinch_thumb_blend` | - | 0.0 | 捏合时拇指直接控制比例 |
| `lp_filter_v2.alpha` | - | 1.0 | v2 小变化 EMA 系数；当前等价于直通 |
| `lp_filter_v2.bypass_ratio` | - | 0.008 | 大变化旁路阈值占关节范围的比例 |

## 11. 适用场景建议

### 更适合使用 v1 的情况

- 机器人关节结构较通用，不具备清晰的单指主动控制通道。
- 更关注整体手型连续性，而不是严格的指尖接触距离。
- 希望使用更少的机器人专用参数。
- 输入动作可能包含多指同时靠近，不希望由唯一伙伴状态机进行筛选。

### 更适合使用 v2 的情况

- 机械手存在少量主动通道和大量 mimic joint，例如 Inspire RH56。
- 需要食指、中指、无名指和小指具有更明显的独立跟随能力。
- 需要拇指与指定伙伴形成可靠的双指接触。
- 对非捏合状态的实时响应和计算开销较敏感。
- 可以针对具体机械手标定直接控制范围和可达接触距离。

## 12. v2 的限制与注意事项

1. v2 当前只选择一个接触伙伴，不直接支持多指同时接触优化。
2. `direct_control_v2` 依赖主动关节名称；换机器人时需要重新配置 `joint_names` 或补充对应映射。
3. `target_distances_cm` 是机械手相关参数，不能直接在不同手型之间复制。
4. 接触目标仍受关节上下界和机械结构限制；目标距离不可达时，SLSQP 只能返回边界内的最接近姿态。
5. 当前状态没有多帧锁存或退出滞后，捏合条件失效后下一帧即回到直接控制。
6. v2 的非捏合直控路径不计算 v1 全手 IK，因此其行为主要由弯曲角标定和拇指横向比例标定决定。
7. 当前 v2 配置的滤波 `alpha` 为 1.0，实际不抑制微抖；是否降低该值应根据真实设备延迟和噪声测量决定。

## 13. 最终总结

从代码结构上看，v2 是 v1 的兼容扩展：

$$
\boxed{
\text{v2}
=
\text{v1 基础 IK}
+
\text{独立通道直控}
+
\text{唯一伙伴选择}
+
\text{显式接触损失}
+
\text{独立关节滤波}
}
$$

两者最本质的区别不在于求解器，而在于控制任务的拆分方式：v1 始终求解一个全手连续优化问题；v2 则让普通手指跟随走直接通道，只把明确的双指接触交给带显式距离约束的优化器。
