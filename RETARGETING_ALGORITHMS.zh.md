# AnyDexRetarget 重定向算法与数学模型

本文档根据当前仓库实际实现整理，说明 AnyDexRetarget 如何把 21 个手部关键点重定向为机器人灵巧手关节角，并给出代码中真正使用的目标函数、梯度和滤波公式。

## 1. 结论概览

AnyDexRetarget 使用的不是闭式解析逆运动学，也不是神经网络，而是以下算法组合：

1. 基于 SVD 的手掌坐标系估计。
2. 人手骨架关键向量与分段比例映射。
3. 基于 Pinocchio 正运动学和位置 Jacobian 的非线性逆运动学。
4. Huber 鲁棒损失。
5. 捏合距离驱动的自适应多目标混合。
6. 手写解析梯度。
7. NLopt SLSQP 有界约束优化。
8. 上一帧 warm start 与关节变化正则化。
9. URDF mimic joint 约束与梯度映射。
10. 一阶低通滤波。

项目包含两个可选择的重定向优化器：

| 优化器 | 状态 | 核心方法 |
|---|---|---|
| `AdaptiveOptimizerAnalytical` | 默认 | 在捏合目标和全手姿态目标之间自适应混合 |
| `KeyVectorOptimizer` | 可选 | 匹配 YAML 中显式定义的人手/机器人关键向量 |

仿真和真机入口的 `--optimizer` 默认值均为 `adaptive`。配置工厂只接受以上两个优化器类型。

## 2. 实际调用链

```text
原始 21x3 手部关键点
    -> 腕部平移归一化
    -> SVD 估计手掌坐标系
    -> 左/右手 MANO 坐标变换
    -> 可选 XYZ 欧拉角校正
    -> 构造目标向量
    -> Pinocchio FK 与 Jacobian
    -> SLSQP 求解带关节限位的非线性 IK
    -> 展开 URDF mimic joints
    -> 一阶低通滤波
    -> 机器人关节角 q
```

高层调用入口为 [`Retargeter.retarget`](anydexretarget/retarget.py)，优化器由 [`BaseOptimizer.from_config`](anydexretarget/optimizer/base_optimizer.py) 根据 YAML 创建。

## 3. 数学符号

| 符号 | 含义 |
|---|---|
| $\mathbf p_j$ | 输入的第 $j$ 个人手关键点，单位 m |
| $\mathbf q$ | 机器人完整关节角向量 |
| $\mathbf q_{prev}$ | 上一帧机器人关节角 |
| $\mathbf x_k(\mathbf q)$ | 机器人 link 或附着点 $k$ 的 FK 位置 |
| $J_k(\mathbf q)$ | 点 $k$ 对关节角的位置 Jacobian |
| $N_f$ | 参与优化的手指数，通常为 4 或 5 |
| $\rho_\delta$ | 阈值为 $\delta$ 的 Huber loss |
| $\alpha_i$ | 第 $i$ 根手指的捏合混合权重 |
| $\varepsilon$ | 防止除零的小常数，代码中通常为 $10^{-8}$ |

MediaPipe 关键点索引约定：

| 部位 | 拇指 | 食指 | 中指 | 无名指 | 小指 |
|---|---:|---:|---:|---:|---:|
| MCP/根部 | 1 | 5 | 9 | 13 | 17 |
| PIP | 2 | 6 | 10 | 14 | 18 |
| DIP | 3 | 7 | 11 | 15 | 19 |
| TIP | 4 | 8 | 12 | 16 | 20 |

腕部索引为 0。

## 4. 输入坐标系算法

实现位置：[`anydexretarget/mediapipe.py`](anydexretarget/mediapipe.py)。

### 4.1 腕部平移归一化

首先把所有点平移到腕部原点：

$$
\tilde{\mathbf p}_j
=
\mathbf p_j-\mathbf p_0
$$

因此后续重定向主要处理相对手型，不直接依赖手在世界坐标系中的绝对位置。

### 4.2 SVD 估计手掌平面

取腕部、食指 MCP 和中指 MCP：

$$
P=
\begin{bmatrix}
\tilde{\mathbf p}_0^\top\\
\tilde{\mathbf p}_5^\top\\
\tilde{\mathbf p}_9^\top
\end{bmatrix}
$$

对这些点去中心化：

$$
\bar P=P-\mathbf 1\boldsymbol\mu^\top,
\qquad
\boldsymbol\mu=\frac{1}{3}\sum_{r=1}^{3}P_r
$$

执行奇异值分解：

$$
\bar P=U\Sigma V^\top
$$

最小奇异值对应的右奇异向量作为手掌平面法向量：

$$
\mathbf n=V_{3,:}
$$

### 4.3 构造腕部坐标系

代码先取腕部到中指根部方向：

$$
\mathbf a=\tilde{\mathbf p}_0-\tilde{\mathbf p}_9
$$

将其投影到手掌平面并归一化：

$$
\mathbf x=
\frac{
\mathbf a-(\mathbf a^\top\mathbf n)\mathbf n
}{
\left\|\mathbf a-(\mathbf a^\top\mathbf n)\mathbf n\right\|_2
}
$$

第三个坐标轴为：

$$
\mathbf z=\mathbf x\times\mathbf n
$$

根据食指 MCP 的方向检查坐标系朝向，必要时同时翻转 $\mathbf n$ 和 $\mathbf z$。最终腕部旋转矩阵为：

$$
R_w=
\begin{bmatrix}
\mathbf x & \mathbf n & \mathbf z
\end{bmatrix}
$$

### 4.4 MANO 坐标转换

左右手分别使用固定的坐标变换矩阵 $C_{right}$ 或 $C_{left}$：

$$
\mathbf p'_j
=
\tilde{\mathbf p}_j R_w C_{hand}
$$

若 YAML 设置了 `mediapipe_rotation`，还会施加外旋 XYZ 欧拉角校正：

$$
\mathbf p''_j
=
\mathbf p'_j R_{xyz}^{\top}
$$

## 5. Huber 鲁棒损失

实现位置：[`anydexretarget/optimizer/utils.py`](anydexretarget/optimizer/utils.py)。

两个优化器都使用 Huber loss，以降低异常关键点和深度抖动对 IK 的影响：

$$
\rho_\delta(x)=
\begin{cases}
\dfrac{1}{2}x^2, & |x|\le\delta\\[6pt]
\delta\left(|x|-\dfrac{1}{2}\delta\right), & |x|>\delta
\end{cases}
$$

对应导数为：

$$
\rho'_\delta(x)=
\begin{cases}
x, & |x|\le\delta\\[6pt]
\delta\operatorname{sign}(x), & |x|>\delta
\end{cases}
$$

小误差区域为二次损失，便于精确拟合；大误差区域为线性增长，避免离群点产生过大的梯度。

## 6. 默认算法：AdaptiveOptimizerAnalytical

实现位置：[`anydexretarget/optimizer/analytical_optimizer.py`](anydexretarget/optimizer/analytical_optimizer.py)。

该算法根据捏合距离，在以下两个目标之间逐指混合：

- `TipDirVec`：强调指尖位置和末端方向，适合捏合与接触。
- `FullHandVec`：强调 PIP、DIP、TIP 整条手指链，适合保持自然手型。

这里的 `Analytical` 表示梯度由代码显式推导和组装，不表示存在直接计算关节角的闭式解。

### 6.1 捏合检测与混合权重

对每根非拇指手指 $i$，计算其指尖到拇指指尖的距离，内部转换为 cm：

$$
d_i
=
100\left\|
\mathbf p^{tip}_i-\mathbf p^{tip}_{thumb}
\right\|_2
$$

通过两个距离阈值 $d_{1,i}$ 和 $d_{2,i}$ 构造线性权重：

$$
\alpha_i
=
\operatorname{clip}
\left(
\frac{d_{2,i}-d_i}{d_{2,i}-d_{1,i}+\varepsilon},
0,
\alpha_{max}
\right)
$$

拇指权重取所有非拇指中的最大值：

$$
\alpha_{thumb}
=
\max_{i\ne thumb}\alpha_i
$$

权重含义如下：

$$
d_i\ge d_{2,i}
\quad\Longrightarrow\quad
\alpha_i=0
\quad\text{（全手姿态模式）}
$$

$$
d_i\le d_{1,i}
\quad\Longrightarrow\quad
\alpha_i\approx\alpha_{max}
\quad\text{（捏合模式）}
$$

源码类常量的默认上限为 0.7，意图是在捏合时仍保留少量全手姿态约束；YAML 中的 `retarget.alpha` 可以覆盖该值。仓库多数 adaptive 配置显式使用 `alpha: 1.0`。

### 6.2 分段骨架缩放

人手和机器人手通常具有不同的掌宽与指节长度。项目不是对整根手指使用一个统一比例，而是逐段重建目标链。

对第 $i$ 根手指，定义四个缩放系数：

$$
s_{i,mcp},\quad
s_{i,pip},\quad
s_{i,dip},\quad
s_{i,tip}
$$

腕部到 MCP 的目标向量为：

$$
\mathbf v_{i,mcp}
=
s_{i,mcp}
D
\left(
\mathbf p_{i,mcp}-\mathbf p_w
\right)
$$

其中 $D$ 是可选的掌部横向缩放矩阵。默认情况下 $D=I$。

随后累积生长手指链：

$$
\mathbf t_{i,pip}
=
\mathbf v_{i,mcp}
+
s_{i,pip}
\left(
\mathbf p_{i,pip}-\mathbf p_{i,mcp}
\right)
$$

$$
\mathbf t_{i,dip}
=
\mathbf t_{i,pip}
+
s_{i,dip}
\left(
\mathbf p_{i,dip}-\mathbf p_{i,pip}
\right)
$$

$$
\mathbf t_{i,tip}
=
\mathbf t_{i,dip}
+
s_{i,tip}
\left(
\mathbf p_{i,tip}-\mathbf p_{i,dip}
\right)
$$

这种累积式缩放保留人手掌面内的手指方向，同时独立校正各段长度。

### 6.3 捏合目标额外缩放

当 `pinch_scaling` 不等于 1 时，算法找到当前捏合权重最大的非拇指手指：

$$
k
=
\underset{i\ne thumb}{\arg\max}\;\alpha_i
$$

归一化混合比例为：

$$
\beta
=
\min
\left(
\frac{\alpha_k}{\alpha_{max}+\varepsilon},
1
\right)
$$

统一缩放的腕部到指尖目标为：

$$
\mathbf t^{pinch}_{i,tip}
=
s_{pinch}
\left(
\mathbf p_{i,tip}-\mathbf p_w
\right)
$$

对拇指和捏合伙伴手指进行插值：

$$
\mathbf t'_{i,tip}
=
(1-\beta)\mathbf t_{i,tip}
+
\beta\mathbf t^{pinch}_{i,tip},
\qquad
i\in\{thumb,k\}
$$

这样可以避免逐段比例差异破坏拇指与目标手指的接触关系。

### 6.4 指尖位置损失

机器人第 $i$ 根手指的 origin 到 tip 向量为：

$$
\mathbf r^{pos}_i(\mathbf q)
=
\mathbf x^{tip}_i(\mathbf q)
-
\mathbf x^{origin}_i(\mathbf q)
$$

位置残差为：

$$
\mathbf e^{pos}_i(\mathbf q)
=
\mathbf r^{pos}_i(\mathbf q)
-
\mathbf t_{i,tip}
$$

指尖位置损失为：

$$
L^{pos}_i(\mathbf q)
=
\rho_{\delta_p}
\left(
\left\|\mathbf e^{pos}_i(\mathbf q)\right\|_2
\right)
$$

### 6.5 指尖方向损失

机器人末端指骨的未归一化方向为：

$$
\mathbf v_i(\mathbf q)
=
\mathbf x^{tip}_i(\mathbf q)
-
\mathbf x^{link4}_i(\mathbf q)
$$

归一化方向为：

$$
\mathbf u_i(\mathbf q)
=
\frac{
\mathbf v_i(\mathbf q)
}{
\left\|\mathbf v_i(\mathbf q)\right\|_2+\varepsilon
}
$$

人手目标方向由 DIP 指向 TIP：

$$
\mathbf u_i^*
=
\frac{
\mathbf p_{i,tip}-\mathbf p_{i,dip}
}{
\left\|\mathbf p_{i,tip}-\mathbf p_{i,dip}\right\|_2+\varepsilon
}
$$

方向损失为：

$$
L^{dir}_i(\mathbf q)
=
\rho_{\delta_d}
\left(
\left\|
\mathbf u_i(\mathbf q)-\mathbf u_i^*
\right\|_2
\right)
$$

### 6.6 全手姿态损失

机器人腕部位置记为 $\mathbf x_w(\mathbf q)$。机器人腕部到三个目标点的向量为：

$$
\mathbf r_{i,pip}(\mathbf q)
=
\mathbf x_{i,pip}(\mathbf q)-\mathbf x_w(\mathbf q)
$$

$$
\mathbf r_{i,dip}(\mathbf q)
=
\mathbf x_{i,dip}(\mathbf q)-\mathbf x_w(\mathbf q)
$$

$$
\mathbf r_{i,tip}(\mathbf q)
=
\mathbf x_{i,tip}(\mathbf q)-\mathbf x_w(\mathbf q)
$$

分别与分段缩放目标比较：

$$
\mathbf e_{i,k}(\mathbf q)
=
\mathbf r_{i,k}(\mathbf q)-\mathbf t_{i,k},
\qquad
k\in\{pip,dip,tip\}
$$

全手姿态损失取三个 Huber loss 的平均：

$$
L^{full}_i(\mathbf q)
=
\frac{1}{3}
\sum_{k\in\{pip,dip,tip\}}
\rho_{\delta_p}
\left(
\left\|\mathbf e_{i,k}(\mathbf q)\right\|_2
\right)
$$

### 6.7 帧间正则化

为了降低关节角跳变，目标函数惩罚当前解与上一帧解的差异：

$$
L^{reg}(\mathbf q)
=
\lambda
\left\|
\mathbf q-\mathbf q_{prev}
\right\|_2^2
$$

对应梯度为：

$$
\nabla_{\mathbf q}L^{reg}
=
2\lambda
\left(
\mathbf q-\mathbf q_{prev}
\right)
$$

### 6.8 Adaptive 最终目标函数

每根手指的捏合目标为：

$$
L^{tip}_i(\mathbf q)
=
w_{pos}L^{pos}_i(\mathbf q)
+
w_{dir}L^{dir}_i(\mathbf q)
$$

每根手指的混合目标为：

$$
L_i(\mathbf q)
=
\alpha_iL^{tip}_i(\mathbf q)
+
(1-\alpha_i)w_{full}L^{full}_i(\mathbf q)
$$

最终目标函数为：

$$
\boxed{
L_{adaptive}(\mathbf q)
=
\sum_{i=1}^{N_f}
\left[
\alpha_i
\left(
w_{pos}L^{pos}_i(\mathbf q)
+
w_{dir}L^{dir}_i(\mathbf q)
\right)
+
(1-\alpha_i)
w_{full}L^{full}_i(\mathbf q)
\right]
+
\lambda
\left\|
\mathbf q-\mathbf q_{prev}
\right\|_2^2
}
$$

## 7. 可选算法：KeyVectorOptimizer

实现位置：[`anydexretarget/optimizer/key_vector_optimizer.py`](anydexretarget/optimizer/key_vector_optimizer.py)。

该算法允许在 YAML 中显式定义任意数量的关键向量。每个条目包含：

- 机器人 origin link 和 task link。
- 可选的 link 局部坐标偏移。
- 人手 origin keypoint 和 task keypoint。
- 人手向量缩放系数。

### 7.1 人手目标向量

第 $i$ 个目标向量为：

$$
\mathbf t_i
=
100s_i
\left(
\mathbf p_{task(i)}-\mathbf p_{origin(i)}
\right)
$$

其中 100 用于把 m 转换为 cm。

### 7.2 机器人关键向量

机器人对应向量为：

$$
\mathbf r_i(\mathbf q)
=
\mathbf x_{task(i)}(\mathbf q)
-
\mathbf x_{origin(i)}(\mathbf q)
$$

残差为：

$$
\mathbf e_i(\mathbf q)
=
\mathbf r_i(\mathbf q)-\mathbf t_i
$$

### 7.3 KeyVector 最终目标函数

对全部 $N$ 个关键向量取平均 Huber loss，并加入帧间正则化：

$$
\boxed{
L_{KV}(\mathbf q)
=
\frac{1}{N}
\sum_{i=1}^{N}
\rho_\delta
\left(
\left\|\mathbf e_i(\mathbf q)\right\|_2
\right)
+
\lambda
\left\|
\mathbf q-\mathbf q_{prev}
\right\|_2^2
}
$$

这个优化器更通用、目标结构更简单，但没有捏合检测与自适应目标切换。

## 8. 解析梯度

实现位置：[`anydexretarget/optimizer/analytical_optimizer.py`](anydexretarget/optimizer/analytical_optimizer.py) 和 [`anydexretarget/optimizer/key_vector_optimizer.py`](anydexretarget/optimizer/key_vector_optimizer.py)。

对于一般向量残差：

$$
\mathbf e(\mathbf q)
=
\mathbf r(\mathbf q)-\mathbf t
$$

令：

$$
d(\mathbf q)=\left\|\mathbf e(\mathbf q)\right\|_2
$$

则 Huber 距离损失的关节梯度为：

$$
\nabla_{\mathbf q}\rho_\delta(d)
=
\rho'_\delta(d)
\frac{\mathbf e^\top}{d+\varepsilon}
J_{\mathbf e}
$$

若机器人向量由 task 点减去 origin 点构成，则：

$$
J_{\mathbf e}
=
J_{task}-J_{origin}
$$

### 8.1 归一化方向的 Jacobian

对于：

$$
\mathbf u
=
\frac{\mathbf v}{\left\|\mathbf v\right\|_2}
$$

其对 $\mathbf v$ 的 Jacobian 为：

$$
\frac{\partial\mathbf u}{\partial\mathbf v}
=
\frac{
I-\mathbf u\mathbf u^\top
}{
\left\|\mathbf v\right\|_2+\varepsilon
}
$$

因此指尖方向对关节角的 Jacobian 为：

$$
J_{dir}
=
\frac{
I-\mathbf u\mathbf u^\top
}{
\left\|\mathbf v\right\|_2+\varepsilon
}
\left(
J_{tip}-J_{link4}
\right)
$$

Pinocchio 负责计算 FK 和各附着点的位置 Jacobian，项目代码再按照以上链式法则组装完整目标梯度。

## 9. SLSQP 有界约束优化

实现位置：[`anydexretarget/optimizer/base_optimizer.py`](anydexretarget/optimizer/base_optimizer.py)。

项目使用 NLopt 的 `LD_SLSQP` 求解：

$$
\mathbf q^*
=
\underset{\mathbf q}{\arg\min}\;L(\mathbf q)
$$

关节角必须满足 URDF 提供的上下限：

$$
\mathbf q_{min}
\le
\mathbf q
\le
\mathbf q_{max}
$$

当前代码设置为：

| 参数 | 数值 |
|---|---:|
| 优化算法 | `nlopt.LD_SLSQP` |
| 最大目标函数计算次数 | 50 |
| 绝对目标函数容差 | $10^{-4}$ |
| 初始值 | 上一帧解、neutral pose 或关节范围中点 |

上一帧结果既作为优化初始值，也作为正则项参考值，这使连续帧通常只需要在上一帧附近进行局部更新。

## 10. Mimic joint 约束

项目从 URDF 的 `<mimic>` 标签读取从属关节关系。对于 mimic joint $q_m$：

$$
q_m
=
a q_s+b
$$

优化器只优化独立关节 $q_s$，FK 前再展开完整关节向量。

梯度通过链式法则映射回源关节：

$$
\frac{\partial L}{\partial q_s}
\mathrel{+}=
a
\frac{\partial L}{\partial q_m}
$$

这保证优化结果符合机器人 URDF 中的机械耦合关系。

## 11. 输出低通滤波

实现位置：[`anydexretarget/optimizer/utils.py`](anydexretarget/optimizer/utils.py)。

优化后的关节角经过一阶指数低通滤波：

$$
\mathbf y_t
=
\mathbf y_{t-1}
+
\gamma
\left(
\mathbf q_t-\mathbf y_{t-1}
\right)
$$

等价形式为：

$$
\boxed{
\mathbf y_t
=
(1-\gamma)\mathbf y_{t-1}
+
\gamma\mathbf q_t
}
$$

其中：

- $\mathbf q_t$ 是当前帧优化结果。
- $\mathbf y_t$ 是当前帧滤波输出。
- $\gamma$ 对应 YAML 中的 `lp_alpha`。
- $\gamma$ 越小，输出越平滑，但响应延迟越明显。

部分输入设备还在关键点进入重定向器之前使用指数加权多帧平滑，但该输入级平滑并不是所有设备的统一必经路径。

## 12. 两个优化器的比较

| 维度 | AdaptiveOptimizerAnalytical | KeyVectorOptimizer |
|---|---|---|
| 默认使用 | 是 | 否 |
| 关键点目标 | 固定的 TIP、DIP、PIP 结构 | YAML 任意定义 |
| 捏合检测 | 有 | 无 |
| 目标混合 | TipDirVec 与 FullHandVec | 单一关键向量目标 |
| 分段缩放 | 有 | 每个向量单独 scale |
| Huber loss | 有 | 有 |
| 解析梯度 | 有 | 有 |
| SLSQP | 有 | 有 |
| 帧间正则 | 有 | 有 |
| 适合场景 | 实时遥操作、捏合与自然手型兼顾 | 快速适配新机器人或自定义向量约束 |

## 13. 代表性配置参数

以 `example/config/adaptive/pico4/pico4_wuji_hand.yaml` 为例：

| 参数 | 含义 | 配置值 |
|---|---|---:|
| `huber_delta` | 位置 Huber 阈值，cm | 2.0 |
| `huber_delta_dir` | 方向 Huber 阈值 | 0.5 |
| `norm_delta` | 帧间关节变化权重 | 0.04 |
| `w_pos` | 指尖位置权重 | 5.0 |
| `w_dir` | 指尖方向权重 | 1.0 |
| `w_full_hand` | 全手姿态权重 | 1.0 |
| `d1` | 进入强捏合区的距离，cm | 2.0 |
| `d2` | 离开捏合区的距离，cm | 4.0 |
| `pinch_scaling` | 捏合时腕部到指尖缩放 | 1.2291 |
| `alpha` | 捏合权重上限 | 1.0 |

不同机器人和输入设备使用不同的 `segment_scaling`、`pinch_scaling` 和坐标旋转参数。这些参数属于机器人形态与输入设备标定，不是求解器本身的固定常量。

## 14. 当前实现边界

以下结论来自当前源码，而不是配置注释：

1. `project_tip_dir` 会从 YAML 读取，但当前没有进入目标计算，因此目前不生效。
2. `finger_root_vectors` 会在初始化时计算，但当前 adaptive 损失没有使用该变量。
3. `AdaptiveOptimizerAnalytical` 的名称容易被误解；它仍然是迭代式非线性优化，只是采用解析梯度。
4. SLSQP 求得的是以上加权目标在当前初始值附近的数值解，不保证全局最优。
5. 输入坐标系估计依赖腕部、食指 MCP 和中指 MCP 不退化；当三点接近共线或检测严重异常时，SVD 坐标系可能不稳定。
6. 优化失败时，代码返回本次优化初始值，而不是抛出错误终止实时控制。

## 15. 一句话总结

AnyDexRetarget 的默认重定向算法可以概括为：

> 使用 SVD 对齐人手坐标系，按手指分段缩放目标骨架，以捏合距离自适应混合指尖接触目标和全手姿态目标，再利用 Pinocchio 解析 Jacobian 与 NLopt SLSQP 在关节限位内求解，并通过帧间正则化和低通滤波保证实时输出连续性。

