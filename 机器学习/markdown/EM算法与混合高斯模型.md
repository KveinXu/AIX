### EM算法与混合高斯模型

***

【参考资料】

吴恩达	CS229

周志华	《机器学习》

[详解EM算法与混合高斯模型(Gaussian mixture model, GMM)](https://blog.csdn.net/lin_limin/article/details/81048411)



#### 1. 混合高斯模型

首先回顾一下一维高斯分布的概率密度函数：
$$
f ( x ) = \frac { 1 } { \sqrt { 2 \pi } \sigma } \exp \left( - \frac { ( x - \mu ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right)
$$
其中$\mu$和$\sigma^2$分别为均值和方差。

多维变量$X = \left( x _ { 1 } , x _ { 2 } , \ldots  ,x _ { n } \right)$的联合概率密度函数为：
$$
f ( X ) = \frac { 1 } { ( 2 \pi ) ^ {n / 2 } | \Sigma | ^ { 1 / 2 } } \exp \left[ - \frac { 1 } { 2 } ( X - u ) ^ { T } \Sigma ^ { - 1 } ( X - u ) \right]
$$
其中，$n$为变量维数，$u=(u_1, u_2, ..., u_n)$为均值，$\Sigma​$为协方差矩阵。

混合高斯模型（Gaussian mixture model，GMM）就是多个高斯分布的合成。假设组成GMM的高斯分布有$K​$个，则GMM的概率密度函数如下：
$$
p ( x ) = \sum _ { j = 1 } ^ { K } p ( j ) p ( x | j) = \sum _ { j = 1 } ^ { K } \phi _ { j } N ( x | u _ { j } , \Sigma _ { j } )
$$
其中$p ( x | j ) = N ( x | u _ { j } , \Sigma _ { j } )$是第$j$个高斯分布的概率密度函数，可以看成是选定第$j$个高斯分布后，该高斯分布产生$x$的概率；$p(j)=\phi_j$是第$j$个高斯分布的权重，也可以视为选择第$j$个高斯分布的概率，并且满足$\sum _ { j = 1 } ^ { K } \phi _ { j } = 1$。

所以，混合高斯模型并不是什么新奇的东西，它的本质就是融合几个单高斯模型，来使得模型更加复杂，从而产生更复杂的样本。理论上，如果某个混合高斯模型融合的高斯模型个数足够多，它们之间的权重设定得足够合理，这个混合模型可以拟合任意分布的样本。

GMM的图例如下

一维的情形：

<img src="assets/1551794561095.png" style="zoom:80%">

二维空间中三个高斯分布的混合：

<img src="assets/1551794604208.png" style="zoom:80%">



#### 2. EM算法估计参数

GMM的参数有各个高斯成分对应的均值$\mu$与协方差$\Sigma$，以及它们的权重$\phi$。在估计GMM模型参数时的一个问题是，我们无法确定观测到的样本$x$具体是由哪一个高斯成分产生的。换句话说，“哪一个高斯分布”是一个隐变量，我们把它用$z$来表示。这时候，就需要用到EM算法来估计GMM的参数。

对于特定的样本$i$，我们假设$z$服从一个参数为$\phi$的多项式分布，并且$\phi_{j} \geq 0$，$\sum_{j=1}^{k} \phi_{j}=1$，其中$j$表示单个高斯成分的序号。于是有：
$$
p\left(z^{(i)}=j\right)=\phi_j
$$
表示选择第$j$个高斯分布的概率是$\phi_j$，并且我们有：
$$
x^{(i)}\left|z^{(i)}=j \sim \mathcal{N}\left(\mu_{j}, \Sigma_{j}\right)\right.
$$
即$x^{(i)}$是由这第$j$个高斯分布产生的。所以，总的似然函数为：
$$
\begin{aligned} \ell(\phi, \mu, \Sigma) &=\sum_{i=1}^{m} \log p\left(x^{(i)} ; \phi, \mu, \Sigma\right) \\ &=\sum_{i=1}^{m} \log \sum_{z^{(i)}=1}^{k} p\left(x^{(i)} | z^{(i)} ; \mu, \Sigma\right) p\left(z^{(i)} ; \phi\right) \end{aligned}
$$
所以，我们需要估计的参数是$\phi_{j}$，$\mu_{j}$，$\Sigma_{j}$。接下来就是调用EM算法的步骤，首先初始化一组$\phi_{j}$，$\mu_{j}$，$\Sigma_{j}$。然后在**E-Step**中，我们利用贝叶斯公式，计算$z^{(i)}$的后验概率：
$$
\begin {eqnarray} 
w_{j}^{(i)}&=&Q_{i}\left(z^{(i)}=j\right)=P\left(z^{(i)}=j | x^{(i)} ; \phi, \mu, \Sigma\right) \\
&=&\frac{p\left(x^{(i)} | z^{(i)}=j ; \mu, \Sigma\right) p\left(z^{(i)}=j ; \phi\right)}{\sum_{l=1}^{k} p\left(x^{(i)} | z^{(i)}=l ; \mu, \Sigma\right) p\left(z^{(i)}=l ; \phi\right)}
 \end {eqnarray}
$$
$w_j^{(i)}$就代表了我们对$z^{(i)}$的选择的一种“猜测”，并且因为计算的是概率，所以这是一种“soft”的、不确定的猜测。

接下来，利用我们对$z^{(i)}$的猜测，在**M-Step**中更新参数，也就是说，我们要最大化下面的似然函数：
$$
\begin{aligned} \sum_{i=1}^{m} & \sum_{z} Q_{i}\left(z^{(i)}\right) \log \frac{p\left(x^{(i)}, z^{(i)} ; \phi, \mu, \Sigma\right)}{Q_{i}\left(z^{(i)}\right)} \\ &=\sum_{i=1}^{m} \sum_{j=1}^{k} Q_{i}\left(z^{(i)}=j\right) \log \frac{p\left(x^{(i)} | z^{(i)}=j ; \mu, \Sigma\right) p\left(z^{(i)}=j ; \phi\right)}{Q_{i}\left(z^{(i)}=j\right)} \\ &=\sum_{i=1}^{m} \sum_{j=1}^{k} w_{j}^{(i)} \log \frac{\frac{1}{(2 \pi)^{n / 2}\left|\Sigma_{j}\right|^{1 / 2}} \exp \left(-\frac{1}{2}\left(x^{(i)}-\mu_{j}\right)^{T} \Sigma_{j}^{-1}\left(x^{(i)}-\mu_{j}\right)\right) \cdot \phi_{j}}{w_{j}^{(i)}} \end{aligned}
$$
为了更新参数$\mu_j$，我们把上面的似然函数对$\mu_j$求导：
$$
\begin{eqnarray} 
\nabla_{\mu_{j}} & \sum_{i=1}^{m} &\sum_{j=1}^{k} w_{j}^{(i)} \log \frac{\frac{1}{(2 \pi)^{n / 2}\left|\Sigma_{j}\right|^{1 / 2}} \exp \left(-\frac{1}{2}\left(x^{(i)}-\mu_{j}\right)^{T} \Sigma_{j}^{-1}\left(x^{(i)}-\mu_{j}\right)\right) \cdot \phi_{j}}{w_{j}^{(i)}} \\ &=&-\nabla_{\mu_{j}} \sum_{i=1}^{m} \sum_{j=1}^{k} w_{j}^{(i)} \frac{1}{2}\left(x^{(i)}-\mu_{j}\right)^{T} \Sigma_{j}^{-1}\left(x^{(i)}-\mu_{j}\right) \\
&=&\sum_{i=1}^{m} w_{j}^{(i)}\left(\Sigma_{j}^{-1} x^{(i)}-\Sigma_{j}^{-1} \mu_{j}\right)
\end{eqnarray}
$$
令其等于0，可得
$$
\mu_{j} :=\frac{\sum_{i=1}^{m} w_{j}^{(i)} x^{(i)}}{\sum_{i=1}^{m} w_{j}^{(i)}}
$$
即各混合成分的均值可通过样本加权平均来估计，样本权重是属于该成分的后验概率。

接下来推导$\phi_j$的更新规则，在似然函数中，与$\phi_{j}$有关的项为
$$
\sum_{i=1}^{m} \sum_{j=1}^{k} w_{j}^{(i)} \log \phi_{j}
$$
但是，由于$\phi_{j}$是一个概率，所以它还需要满足$\sum_{j=1}^{k} \phi_{j}=1$的约束条件，因此构造拉格朗日算子为
$$
\mathcal{L}(\phi)=\sum_{i=1}^{m} \sum_{j=1}^{k} w_{j}^{(i)} \log \phi_{j}+\beta\left(\sum_{j=1}^{k} \phi_{j}-1\right)
$$
其中$\beta$是拉格朗日乘子。对上式求导，可得
$$
\frac{\partial}{\partial \phi_{j}} \mathcal{L}(\phi)=\sum_{i=1}^{m} \frac{w_{j}^{(i)}}{\phi_{j}}+\beta
$$
将其置零，可得
$$
\phi_{j}=\frac{\sum_{i=1}^{m} w_{j}^{(i)}}{-\beta}
$$
利用约束$\sum_{j} \phi_{j}=1$，我们可以将上式改写为：
$$
-\beta=\sum_{i=1}^{m} \sum_{j=1}^{k} w_{j}^{(i)}=\sum_{i=1}^{m} 1=m
$$
这里利用了$\sum_{j} w_{j}^{(i)}=1$。

于是$\phi_{j}$的更新就可以表示为
$$
\phi_{j} :=\frac{1}{m} \sum_{i=1}^{m} w_{j}^{(i)}
$$
即每个高斯成分的混合系数由样本属于该成分的平均后验概率确定。

同理，可以得到$\Sigma_j$的更新规则。

各个参数的更新规则总结如下：
$$
\begin{aligned} \phi_{j} & :=\frac{1}{m} \sum_{i=1}^{m} w_{j}^{(i)} \\ \mu_{j} & :=\frac{\sum_{i=1}^{m} w_{j}^{(i)} x^{(i)}}{\sum_{i=1}^{m} w_{j}^{(i)}} \\ \Sigma_{j} & :=\frac{\sum_{i=1}^{m} w_{j}^{(i)}\left(x^{(i)}-\mu_{j}\right)\left(x^{(i)}-\mu_{j}\right)^{T}}{\sum_{i=1}^{m} w_{j}^{(i)}} \end{aligned}
$$


