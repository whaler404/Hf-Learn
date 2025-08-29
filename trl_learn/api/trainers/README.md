### DPO

dpo 绕过复杂的强化学习，直接用简单的监督学习来完成模型对齐，不需要打分器，直接告诉模型，答案 A 比答案 B 好，写出答案 A 的概率应该更高，核心在于设计损失函数，以“让偏好答案的概率高于非偏好答案”为监督学习目标

$$
\mathcal{L}_{DPO}=-\mathbb{E}_{(x,y_w,y_l)\sim \mathcal{D}}
[\log{\sigma(
\beta\log\frac{\pi_\theta(y_w\vert x)}{\pi_\text{ref}(y_w\vert x)}-
\beta\log\frac{\pi_\theta(y_l\vert x)}{\pi_\text{ref}(y_l\vert x)}-
)}]
$$

$y_w$ 表示 win ，$y_l$ 表示 lose ，分别表示 chosed 和 rejected 的样本

### GRPO

GRPO（group relative policy optimization） 和 PPO 都是异策略算法，都使用重要性采样，复用旧策略的样本来调整新策略的改进方向，区别在于优势函数 $A$ 的设计

$$
\mathcal{J}(\theta)=\mathbb{E}_{q\sim P(Q),\{o_i\}\sim \pi_{\theta_\text{old}}(O\vert q)}\\
\frac{1}{G}\sum_{i=1}^G(
\min(\frac{\pi_\theta(o_i\vert q)}{\pi_{\theta_\text{old}}(o_i\vert q)}A_i,
\text{clip}(\frac{\pi_\theta(o_i\vert q)}{\pi_{\theta_\text{old}}(o_i\vert q)},1-\varepsilon,1+\varepsilon)
)-
\beta\mathbb{D}_\text{KL}(\pi_\theta\parallel\pi_{\theta_\text{old}})
)
$$

其中优势函数为

$$
A_i=\frac{r_i-\text{mean}(r_1,\dots,r_G)}{\text{std}(r_1,\dots,r_G)}
$$

只使用了上一步自己的输出，基于规则给出输出的奖励信号，并使用来组内样本的奖励计算基线，无需 critic 模型的参与，降低计算成本

