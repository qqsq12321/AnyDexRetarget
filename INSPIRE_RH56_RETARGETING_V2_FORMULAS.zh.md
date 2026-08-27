# Inspire RH56DFTP-2R 重定向算法 v2 公式说明

## 1. 文档范围

本文档单独说明 Pico 4 右手关键点到 Inspire Robots `RH56DFTP-2R` 右手的 v2 重定向算法，重点给出当前代码实际使用的数学公式、状态切换、参数和机械可达边界。

对应实现：

- `anydexretarget/optimizer/analytical_optimizer_v2.py`
- `anydexretarget/optimizer/filter_v2.py`
- `anydexretarget/retarget.py`
- `example/config/adaptive/pico4/pico4_inspire_hand_v2.yaml`

v2 不替换原 v1。它在 `AdaptiveOptimizerAnalytical` 基础上增加两状态混合控制：

当前 v2 优化器类名为 `AdaptiveOptimizerAnalyticalV2`。

1. 非捏合时，六个主动通道直接独立跟随人手。
2. 检测到唯一、明确的双指捏合时，只让拇指和目标手指进入接触优化。
3. 其他手指继续独立控制，避免一根手指动作带动整只手。
4. 松开捏合后，下一帧立即恢复六通道直接控制。

## 2. RH56 六个主动通道

RH56 的控制通道顺序为：

```text
[pinky, ring, middle, index, thumb_bend, thumb_rotation]
```

项目 URDF 中六个独立关节为：

| 功能 | URDF 独立关节 | 完整 qpos 索引 |
|---|---|---:|
| 食指弯曲 | `index_proximal_joint` | 0 |
| 中指弯曲 | `middle_proximal_joint` | 2 |
| 小指弯曲 | `pinky_proximal_joint` | 4 |
| 无名指弯曲 | `ring_proximal_joint` | 6 |
| 拇指旋转 | `thumb_proximal_yaw_joint` | 8 |
| 拇指弯曲 | `thumb_proximal_pitch_joint` | 9 |

设六个主动变量组成向量：

$$
\mathbf q_a=
\begin{bmatrix}
q_{index} &
q_{middle} &
q_{pinky} &
q_{ring} &
q_{thumb,yaw} &
q_{thumb,pitch}
\end{bmatrix}^{\mathsf T}
$$

其余关节由 URDF mimic 关系重建。对任一 mimic 关节：

$$
q_m=a_m q_s+b_m
$$

其中 $q_s$ 是源主动关节，$a_m$ 和 $b_m$ 分别是 URDF 中的 `multiplier` 和 `offset`。

## 3. 符号约定

| 符号 | 含义 |
|---|---|
| $\mathbf p_k\in\mathbb R^3$ | 第 $k$ 个人手关键点，单位 m |
| $\mathbf q$ | RH56 完整 URDF 关节向量 |
| $\mathbf q^{direct}$ | 六通道直接控制目标展开后的完整关节向量 |
| $\mathbf q^{ik}$ | v1 IK 与 v2 接触项联合优化得到的关节向量 |
| $\mathbf x_i(\mathbf q)$ | 机器人第 $i$ 根手指指尖 FK 位置 |
| $J_i(\mathbf q)$ | 第 $i$ 根手指指尖的位置 Jacobian |
| $d_i^h$ | 人手拇指与第 $i$ 根手指的指尖距离 |
| $d_i^r$ | 机器人拇指与第 $i$ 根手指的指尖距离 |
| $\alpha_i$ | v1 捏合权重 |
| $\beta_i$ | v2 接触激活强度 |
| $\rho_\delta$ | Huber 鲁棒损失 |
| $\varepsilon$ | 防止除零的小常数，代码中为 $10^{-8}$ 量级 |

MediaPipe 关键点索引：

| 手指 | MCP | PIP | DIP | TIP |
|---|---:|---:|---:|---:|
| 拇指 | 1 | 2 | 3 | 4 |
| 食指 | 5 | 6 | 7 | 8 |
| 中指 | 9 | 10 | 11 | 12 |
| 无名指 | 13 | 14 | 15 | 16 |
| 小指 | 17 | 18 | 19 | 20 |

腕部关键点索引为 0。

## 4. v2 总体状态机

令候选接触伙伴集合为：

$$
\mathcal F=
\{index,middle,ring,pinky\}
$$

v2 每帧先计算所有候选手指的捏合权重，再选择当前最大候选：

$$
i^*=\underset{i\in\mathcal F}{\arg\max}\;\alpha_i
$$

第二大候选记为 $i^{(2)}$。

当前状态分为：

$$
S_t=
\begin{cases}
DIRECT, & \text{没有唯一且明确的捏合伙伴}\\[4pt]
PINCH(i^*), & \text{拇指与 }i^*\text{ 形成明确捏合}
\end{cases}
$$

输出规则为：

$$
\mathbf q_t=
\begin{cases}
\mathbf q_t^{direct}, & S_t=DIRECT\\[4pt]
\operatorname{Mix}
\left(
\mathbf q_t^{ik},
\mathbf q_t^{direct},
i^*
\right), & S_t=PINCH(i^*)
\end{cases}
$$

当前配置没有退出滞后或多帧锁存。因此一旦捏合条件不再成立，下一帧直接返回 `DIRECT`。

## 5. 非拇指手指独立控制

### 5.1 两段弯曲角

对食指、中指、无名指和小指，定义三段方向：

$$
\mathbf v_1=\mathbf p_{PIP}-\mathbf p_{MCP}
$$

$$
\mathbf v_2=\mathbf p_{DIP}-\mathbf p_{PIP}
$$

$$
\mathbf v_3=\mathbf p_{TIP}-\mathbf p_{DIP}
$$

任意两段向量的夹角为：

$$
\theta(\mathbf a,\mathbf b)=
\arccos
\left(
\operatorname{clip}
\left(
\frac{\mathbf a^{\mathsf T}\mathbf b}
{(\lVert\mathbf a\rVert_2+\varepsilon)
(\lVert\mathbf b\rVert_2+\varepsilon)},
-1,
1
\right)
\right)
$$

代码中先分别归一化两段向量。等价的实际实现为：

$$
\theta(\mathbf a,\mathbf b)=
\arccos
\left(
\operatorname{clip}
\left(
\frac{\mathbf a}{\lVert\mathbf a\rVert_2+\varepsilon}
\cdot
\frac{\mathbf b}{\lVert\mathbf b\rVert_2+\varepsilon},
-1,
1
\right)
\right)
$$

该手指的总弯曲量为：

$$
c_i=
\theta(\mathbf v_1,\mathbf v_2)
+
\theta(\mathbf v_2,\mathbf v_3)
$$

### 5.2 归一化到关节范围

设标定的张开和闭合弯曲角分别为 $c_{open}$、$c_{closed}$：

$$
r_i=
\operatorname{clip}
\left(
\frac{c_i-c_{open}}
{c_{closed}-c_{open}+\varepsilon},
0,
1
\right)
$$

再映射到该 RH56 主动关节的上下限：

$$
q_i^{direct}=
q_{i,min}
+
r_i
\left(
q_{i,max}-q_{i,min}
\right)
$$

当前参数为：

$$
c_{open}=0.1\ \text{rad}
$$

$$
c_{closed}=2.8\ \text{rad}
$$

四根非拇指手指分别计算自己的 $c_i$，没有共享平均值，因此单独弯曲一根手指时不会主动改变其他三根手指的直接目标。

## 6. 拇指独立控制

RH56 拇指有两个主动通道：横向旋转和弯曲。

### 6.1 拇指横向旋转

定义食指根部到小指根部的掌宽向量：

$$
\mathbf l=\mathbf p_{17}-\mathbf p_5
$$

掌宽为：

$$
w_p=\lVert\mathbf l\rVert_2
$$

掌横向单位向量为：

$$
\hat{\mathbf e}_{lat}=
\frac{\mathbf l}{w_p+\varepsilon}
$$

用掌宽归一化拇指指尖的横向位置：

$$
h_{thumb}=
\frac{
(\mathbf p_4-\mathbf p_0)^{\mathsf T}
\hat{\mathbf e}_{lat}
}{w_p+\varepsilon}
$$

将其归一化为拇指对掌程度：

$$
r_{yaw}=
\operatorname{clip}
\left(
\frac{h_{thumb}-h_{open}}
{h_{opposed}-h_{open}+\varepsilon},
0,
1
\right)
$$

拇指旋转关节目标为：

$$
q_{thumb,yaw}^{direct}=
q_{yaw,min}
+
r_{yaw}
\left(
q_{yaw,max}-q_{yaw,min}
\right)
$$

当前标定参数为：

$$
h_{open}=-2.05
$$

$$
h_{opposed}=0.1
$$

### 6.2 拇指弯曲

拇指使用关键点 1、2、3、4 构造三段向量：

$$
\mathbf t_1=\mathbf p_2-\mathbf p_1
$$

$$
\mathbf t_2=\mathbf p_3-\mathbf p_2
$$

$$
\mathbf t_3=\mathbf p_4-\mathbf p_3
$$

拇指弯曲量为：

$$
c_{thumb}=
\theta(\mathbf t_1,\mathbf t_2)
+
\theta(\mathbf t_2,\mathbf t_3)
$$

归一化系数为：

$$
r_{pitch}=
\operatorname{clip}
\left(
\frac{c_{thumb}-c_{thumb,open}}
{c_{thumb,closed}-c_{thumb,open}+\varepsilon},
0,
1
\right)
$$

拇指弯曲关节目标为：

$$
q_{thumb,pitch}^{direct}=
q_{pitch,min}
+
r_{pitch}
\left(
q_{pitch,max}-q_{pitch,min}
\right)
$$

当前标定参数为：

$$
c_{thumb,open}=0.13\ \text{rad}
$$

$$
c_{thumb,closed}=1.69\ \text{rad}
$$

## 7. 明确双指捏合检测

### 7.1 人手指尖距离

对每个非拇指候选 $i$，人手指尖距离转换为 cm：

$$
d_i^h=
100
\left\lVert
\mathbf p_{TIP,i}-\mathbf p_{TIP,thumb}
\right\rVert_2
$$

### 7.2 v1 捏合权重

使用每根手指的近距离阈值 $d_{1,i}$ 和远距离阈值 $d_{2,i}$：

$$
\alpha_i=
\operatorname{clip}
\left(
\frac{d_{2,i}-d_i^h}
{d_{2,i}-d_{1,i}+\varepsilon},
0,
\alpha_{max}
\right)
$$

当前 v2 配置中：

$$
\alpha_{max}=1.0
$$

距离阈值为：

| 手指 | $d_1$ | $d_2$ |
|---|---:|---:|
| 食指 | 2 cm | 8 cm |
| 中指 | 2 cm | 4 cm |
| 无名指 | 2 cm | 4 cm |
| 小指 | 2 cm | 4 cm |

### 7.3 接触激活强度

先归一化最大候选的捏合强度：

$$
\bar\beta_{i^*}=
\operatorname{clip}
\left(
\frac{\alpha_{i^*}}
{\alpha_{max}+\varepsilon},
0,
1
\right)
$$

只有满足以下条件才进入接触模式：

$$
\bar\beta_{i^*}\ge\beta_{min}
$$

当前阈值为：

$$
\beta_{min}=0.8
$$

激活后的连续接触强度为：

$$
\beta_{i^*}=
\operatorname{clip}
\left(
\frac{\bar\beta_{i^*}-\beta_{min}}
{1-\beta_{min}+\varepsilon},
0,
1
\right)
$$

因此在刚达到阈值时，接触项从 0 开始连续增加，而不是瞬间施加完整闭合权重。

### 7.4 多指歧义拒绝

如果第二候选也已达到激活阈值，并且前两名差距过小：

$$
\alpha_{i^{(2)}}\ge\beta_{min}
$$

且：

$$
\alpha_{i^*}-\alpha_{i^{(2)}}<m
$$

则拒绝进入接触模式。

当前优势阈值为：

$$
m=0.15
$$

该规则用于避免握拳、三指同时靠近或关键点抖动时，优化器在多个目标手指之间快速切换。

## 8. 接触目标距离

设配置中的机械目标距离为 $d_{reach,i}$，人手距离缩放系数为 $s_p$。

首先计算缩放后的人手目标，并限制其不能小于机械目标：

$$
d_{scaled,i}=
\max
\left(
s_p d_i^h,
d_{reach,i}
\right)
$$

当前缩放系数为：

$$
s_p=1.3224
$$

接触参考距离随 $\beta_i$ 连续过渡：

$$
d_{target,i}=
(1-\beta_i)d_{scaled,i}
+
\beta_i d_{reach,i}
$$

当前 YAML 对四个候选都配置：

$$
d_{reach,i}=0.2\ \text{cm}=2\ \text{mm}
$$

这个数值是优化目标，不代表所有手指都能达到 2 mm。优化器仍受 RH56 关节上下限和机械结构约束。

## 9. 显式接触损失

### 9.1 机器人相对指尖向量

对当前唯一接触伙伴 $i^*$：

$$
\mathbf r(\mathbf q)=
\mathbf x_{i^*}(\mathbf q)
-
\mathbf x_{thumb}(\mathbf q)
$$

机器人指尖距离为：

$$
d^r(\mathbf q)=
\lVert\mathbf r(\mathbf q)\rVert_2
$$

距离残差为：

$$
e(\mathbf q)=
d^r(\mathbf q)-d_{target,i^*}
$$

### 9.2 Huber 损失

Huber 函数为：

$$
\rho_\delta(e)=
\begin{cases}
\dfrac{1}{2}e^2, & |e|\le\delta\\[6pt]
\delta\left(|e|-\dfrac{1}{2}\delta\right), & |e|>\delta
\end{cases}
$$

其导数为：

$$
\rho_\delta'(e)=
\begin{cases}
e, & |e|\le\delta\\[6pt]
\delta\operatorname{sign}(e), & |e|>\delta
\end{cases}
$$

v2 接触损失为：

$$
L_{contact}(\mathbf q)=
w_c\beta_{i^*}
\rho_{\delta_c}
\left(
e(\mathbf q)
\right)
$$

当前参数为：

$$
w_c=160
$$

$$
\delta_c=0.5\ \text{cm}
$$

总优化目标为：

$$
L_{v2}(\mathbf q)=
L_{v1}(\mathbf q)
+
L_{contact}(\mathbf q)
$$

并满足关节限位：

$$
\mathbf q_{min}
\le
\mathbf q
\le
\mathbf q_{max}
$$

该有界非线性问题继续由 NLopt SLSQP 求解。

## 10. 接触损失解析梯度

机器人相对指尖 Jacobian 为：

$$
J_{rel}(\mathbf q)=
J_{i^*}(\mathbf q)
-
J_{thumb}(\mathbf q)
$$

指尖距离对关节角的导数为：

$$
\frac{\partial d^r}{\partial\mathbf q}=
\frac{\mathbf r^{\mathsf T}}
{\lVert\mathbf r\rVert_2+\varepsilon}
J_{rel}
$$

因此接触损失梯度为：

$$
\nabla_{\mathbf q}L_{contact}=
w_c\beta_{i^*}
\rho_{\delta_c}'(e)
\frac{\mathbf r^{\mathsf T}}
{\lVert\mathbf r\rVert_2+\varepsilon}
J_{rel}
$$

对 mimic 关节，梯度通过链式法则映射回主动源关节：

$$
\frac{\partial L}{\partial q_s}
\mathrel{+}=
a_m
\frac{\partial L}{\partial q_m}
$$

## 11. 捏合状态下的通道混合

设优化结果为 $\mathbf q^{ik}$，直接控制结果为 $\mathbf q^{direct}$。

对当前接触伙伴 $i^*$：

$$
q_{i^*}^{out}=
(1-\lambda_{pinch,f})q_{i^*}^{ik}
+
\lambda_{pinch,f}q_{i^*}^{direct}
$$

当前配置：

$$
\lambda_{pinch,f}=0
$$

因此目标手指完全采用接触优化结果。

对拇指旋转和弯曲通道：

$$
q_{thumb}^{out}=
(1-\lambda_{pinch,t})q_{thumb}^{ik}
+
\lambda_{pinch,t}q_{thumb}^{direct}
$$

当前配置：

$$
\lambda_{pinch,t}=0
$$

因此拇指也完全采用接触优化结果。

对所有非目标手指 $j\ne i^*$：

$$
q_j^{out}=
(1-\lambda_f)q_j^{ik}
+
\lambda_f q_j^{direct}
$$

当前配置：

$$
\lambda_f=1
$$

因此其他手指完全保持独立直接控制，不受当前捏合优化拖动。

在非捏合状态，六个通道均使用直接目标：

$$
\mathbf q^{out}=\mathbf q^{direct}
$$

## 12. v2 自适应滤波

对第 $j$ 个主动通道，设当前输入为 $x_{t,j}$，上一帧输出为 $y_{t-1,j}$。

关节范围为：

$$
R_j=q_{j,max}-q_{j,min}
$$

旁路阈值为：

$$
\tau_j=r_b R_j
$$

当前旁路比例为：

$$
r_b=0.008
$$

滤波公式为：

$$
y_{t,j}=
\begin{cases}
x_{t,j},
& |x_{t,j}-y_{t-1,j}|\ge\tau_j\\[4pt]
x_{t,j},
& j\in\mathcal P\\[4pt]
y_{t-1,j}+\alpha_f(x_{t,j}-y_{t-1,j}),
& \text{其他情况}
\end{cases}
$$

其中 $\mathcal P$ 是永久直通通道集合。当前包括：

$$
\mathcal P=
\{thumb\_yaw,thumb\_pitch\}
$$

当前真机配置还设置：

$$
\alpha_f=1.0
$$

所以当前六通道实际都没有额外 EMA 延迟。保留滤波结构仅用于以后根据抖动数据恢复小信号平滑。

滤波只作用于六个主动变量，之后重新展开 mimic 关节，保证：

$$
q_{m,t}=a_m q_{s,t}+b_m
$$

## 13. 真机控制值映射

项目驱动最终输出顺序为：

$$
\mathbf u=
\begin{bmatrix}
u_{pinky} &
u_{ring} &
u_{middle} &
u_{index} &
u_{thumb,bend} &
u_{thumb,rotation}
\end{bmatrix}^{\mathsf T}
$$

对每个通道，关节角先除以对应最大弧度并裁剪：

$$
z_j=
\operatorname{clip}
\left(
\frac{q_j}{q_{j,max}^{driver}},
0,
1
\right)
$$

RH56 协议中 `1000` 表示张开，`0` 表示闭合，因此近似映射为：

$$
u_j\approx
1000(1-z_j)
$$

当前驱动最大弧度为：

$$
\mathbf q_{max}^{driver}=
\begin{bmatrix}
1.47 & 1.47 & 1.47 & 1.47 & 0.6 & 1.308
\end{bmatrix}^{\mathsf T}
$$

## 14. 机械可达边界

### 14.1 指尖中心距离

当前 URDF 关节限位内的代表性最小指尖中心距离为：

| 对指组合 | v2 最小指尖中心距离 | 结论 |
|---|---:|---|
| 拇指-食指 | 2.64 mm | 可接近真正闭合 |
| 拇指-中指 | 12.41 mm | 已接近当前机构边界 |
| 拇指-无名指 | 32.38 mm | 指尖不可达 |
| 拇指-小指 | 52.61 mm | 指尖不可达 |

### 14.2 全局搜索验证

对无名指和小指，进一步在相关手指弯曲、拇指旋转、拇指弯曲三个主动变量上执行全局差分进化搜索：

$$
\min_{\mathbf q}
\left\lVert
\mathbf x_i(\mathbf q)
-
\mathbf x_{thumb}(\mathbf q)
\right\rVert_2
$$

得到：

| 伙伴 | 全局最小指尖距离 | 近似最小表面距离 |
|---|---:|---:|
| 无名指 | 32.38 mm | 15.75 mm |
| 小指 | 52.61 mm | 33.96 mm |

表面距离使用 URDF 碰撞网格顶点采样计算，是近似值，但结果仍显著大于 0。

因此在当前 URDF、关节限位和网格采样精度下：

$$
\min_{\mathbf q}
d_{surface,ring}(\mathbf q)
\approx15.75\ \text{mm}>0
$$

$$
\min_{\mathbf q}
d_{surface,pinky}(\mathbf q)
\approx33.96\ \text{mm}>0
$$

这意味着继续增大 $w_c$、减小 $d_{reach}$ 或增加 SLSQP 迭代次数，都不能让无名指和小指真正接触拇指。优化器只能把它们推到各自最近的机械姿态。

## 15. 当前参数汇总

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `pinch_scaling` | 1.3224 | 人手捏合距离缩放 |
| `alpha` | 1.0 | v1 捏合权重上限 |
| `contact_v2.weight` | 160.0 | 接触损失权重 |
| `contact_v2.huber_delta` | 0.5 cm | 接触 Huber 阈值 |
| `contact_v2.min_beta` | 0.8 | 接触激活阈值 |
| `contact_v2.dominance_margin` | 0.15 | 唯一伙伴优势阈值 |
| `target_distances_cm.*` | 0.2 cm | 期望接触目标，不是可达承诺 |
| `finger_blend` | 1.0 | 非目标手指完全直接控制 |
| `thumb_blend` | 1.0 | 非捏合时拇指完全直接控制 |
| `pinch_finger_blend` | 0.0 | 捏合伙伴完全采用 IK |
| `pinch_thumb_blend` | 0.0 | 捏合时拇指完全采用 IK |
| `lp_filter_v2.alpha` | 1.0 | 当前无滤波延迟 |
| `lp_filter_v2.bypass_ratio` | 0.008 | 大动作旁路阈值比例 |

## 16. 性能与验证结果

500 帧抽样性能：

| 算法 | Mean | P95 | Max |
|---|---:|---:|---:|
| v1 | 8.035 ms | 13.762 ms | 43.087 ms |
| v2 | 2.267 ms | 16.138 ms | 31.228 ms |

v2 平均耗时下降的主要原因是：

$$
S_t=DIRECT
\quad\Longrightarrow\quad
\text{直接返回六通道目标，不运行 SLSQP}
$$

只有明确捏合帧才执行接触优化，因此 P95 仍包含 SLSQP 求解时间。

最近一次离线回归结果：

```text
25 passed in 0.38s
All checks passed!
git diff --check: passed
```

真机串口控制稳定在约 `62 FPS`。

## 17. 仿真观察方法

启动带 Pico 输入的 RH56 v2 骨架调试：

```bash
cd /home/engram/AnyDexRetarget/example

/home/engram/anaconda3/envs/anydex/bin/python \
  test/debug_skeleton.py \
  --robot inspire \
  --input pico4 \
  --pico4-mode relay \
  --hand right \
  --config config/adaptive/pico4/pico4_inspire_hand_v2.yaml \
  --alpha 0.45
```

颜色含义：

| 颜色 | 含义 |
|---|---|
| 蓝色 | Pico 原始人手骨架 |
| 黄色 | `pinch_scaling` 统一缩放预览 |
| 绿色 | 优化器分段缩放目标 |
| 红色 | RH56 重定向后的 FK 骨架 |

## 18. 结论

v2 已解决的重点是控制逻辑，而不是改变 RH56 的机械自由度：

1. 单独活动任一非拇指手指时，对应主动通道独立跟随。
2. 拇指旋转和弯曲分别由独立的人手特征控制。
3. 只有唯一、明确的双指捏合才进入接触优化。
4. 捏合时只接管拇指和目标手指，其他手指继续独立。
5. 拇指-食指可以接近闭合，拇指-中指只能达到机构边界。
6. 拇指-无名指和拇指-小指不存在真正接触的可达解，属于硬件构型限制，不是继续调高损失权重可以解决的问题。

