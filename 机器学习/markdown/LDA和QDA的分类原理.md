### LDA和QDA的分类原理

***

#### 0. 参考资料

**CSDN** [判别模型：logistic,GDA,QDA（一）](https://blog.csdn.net/G090909/article/details/50197331)

https://esl.hohoweiya.xyz/04-Linear-Methods-for-Classification/4.3-Linear-Discriminant-Analysis/index.html

**scikit-learn官网** https://scikit-learn.org/stable/modules/lda_qda.html#mathematical-formulation-of-the-lda-and-qda-classifiers

吴恩达**CS229**讲义中相关内容

《**ESL**》一书中对应章节



#### 1. LDA

线性判别分析（Linear Discriminant Analysis, LDA）和二次判别分析（Quadratic Discriminant Analysis, QDA）都是分类的生成模型，它们的原理与贝叶斯法则有关。

对于分类问题而言，我们需要求得后验概率$P(y=k|X)​$，其中$k=1,\cdots,K​$代表类别。利用贝叶斯公式可以求得后验概率，即：
$$
P(y=k|X)=\frac{P(X|y=k)P(y=k)}{P(X)}=\frac{P(X|y=k)P(y=k)}{\sum_{l=1}^{K} P(X|y=l)P(y=l)}
$$
对于LDA和QDA，我们假设$P(X|y=k)$服从多元高斯分布，先验概率$P(y=k)=\pi_{k}$，这时又称为高斯判别分析（GDA）。

此时有：
$$
\begin{aligned}
P ( X | y = k ) &= f_k(x)
\\&=\frac { 1 } { ( 2 \pi ) ^ { n / 2 } \left| \Sigma _ { k } \right| ^ { 1 / 2 } } \exp \left( - \frac { 1 } { 2 } \left( X - \mu _ { k } \right) ^ { t } \Sigma _ { k } ^ { - 1 } \left( X - \mu _ { k } \right) \right)
\end{aligned}
$$
其中$n$时特征维度。

假设所有类别都具有相同的协方差矩阵$\Sigma_k=\Sigma$，会导出LDA。在比较两个类别$k$和$l$时，考虑它们的对数比率即可，所以有
$$
\begin{aligned} & \log \frac { \operatorname { Pr } ( y = k | X = x ) } { \operatorname { Pr } ( y = \ell | X = x ) } \\ = & \log \frac { f _ { k } ( x ) } { f _ { \ell } ( x ) } + \log \frac { \pi _ { k } } { \pi _ { \ell } } \\ = & \log \frac { \pi _ { k } } { \pi _ { \ell } } - \frac { 1 } { 2 } \left( \mu _ { k } + \mu _ { \ell } \right) ^ { T } \mathbf { \Sigma } ^ { - 1 } \left( \mu _ { k } - \mu _ { \ell } \right) + x ^ { T } \mathbf { \Sigma } ^ { - 1 } \left( \mu _ { k } - \mu _ { \ell } \right) \end{aligned} \tag{1.1}
$$
这是个关于$x$的线性等式，协方差矩阵相等消除了二次项，这也是名称“线性判别分析”的由来。

根据式（1.1）可以看出线性判别函数为：
$$
\delta _ { k } ( x ) = x ^ { T } \mathbf { \Sigma } ^ { - 1 } \mu _ { k } - \frac { 1 } { 2 } \mu _ { k } ^ { T } \mathbf { \Sigma } ^ { - 1 } \mu _ { k } + \log \pi _ { k }
$$
然后类别$k^*=\operatorname{{argmax}}_{k}\delta_{k}(x)$，这是判别规则的等价描述。

我们可以通过最大后验的方式从数据集中估计参数：

* $\hat { \pi } _ { k } = N _ { k } / N$，其中$N_k$是第$k$类观测值的个数；
* $\hat { \mu } _ { k } = \sum _ { i:y _ { i } = k } x _ { i } / N _ { k }$；
* $\hat { \mathbf { \Sigma } } = \sum _ { k = 1 } ^ { K } \sum _ { i:y _ { i } = k } \left( x _ { i } - \hat { \mu } _ { k } \right) \left( x _ { i } - \hat { \mu } _ { k } \right) ^ { T } / ( N - K )$。

#### 2. QDA

如果我们假设每一类别的协方差矩阵不同，那么式（1.1）中方便的抵消就不会发生，这时就是QDA的情况。

此时判别函数变为：
$$
\begin{aligned} \delta _ { k } ( x ) & = - \frac { 1 } { 2 } \left( x - \mu _ { k } \right) ^ { T } \Sigma _ { k } ^ { - 1 } \left( x - \mu _ { k } \right) - \frac { 1 } { 2 } \log \left| \Sigma _ { k } \right| + \log \pi _ { k } \\ & = - \frac { 1 } { 2 } x ^ { T } \Sigma _ { k } ^ { - 1 } x + x ^ { T } \mathbf { \Sigma } _ { k } ^ { - 1 } \mu _ { k } - \frac { 1 } { 2 } \mu _ { k } ^ { T } \Sigma _ { k } ^ { - 1 } \mu _ { k } - \frac { 1 } { 2 } \log \left| \Sigma _ { k } \right| + \log \pi _ { k } \end{aligned}
$$
这时$\delta_k(x)$是关于$x$的二次函数。每个类别对$k$和$l$的判别边界由二次等式$\left\{ x : \delta _ { k } ( x ) = \delta _ { \ell } ( x ) \right\}$来描述。

#### 3. 与LR的联系

可以证明，LDA的后验概率能被表示成logistic回归的形式。那么应该如何选择LDA和LR呢？

注意到LDA假设数据服从多元高斯分布，而某些不服从高斯分布的数据也可以导出LR的形式。因此，LDA的假设要比LR更强。

当数据确实服从多元高斯分布时，使用LDA会带来更好的效果，并且对数据的利用率更高（即只需要少量数据就能学习的很好）。但当数据不服从高斯分布时，强行使用LDA效果不佳，这是LR的表现会更鲁棒，因此实际操作时，我们常常更偏向于使用LR。