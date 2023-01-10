# RNN

## 应用场景

* 时间序列预测
  * 2, 4, 6, 8, 10 ... 预测偶数
  * 股市波动预测
  * 物体温度曲线预测
* 股票价格预测（富时100指数）
    ![FTSE 100 Index](./ftse100.png)
    $x_t ~ P(x_t|x_{t-1}, ..., x_{1})$
    
    即$x_t$与$x_{t-1}, ..., x_{1}$有关

## 训练方法

* 自回归模型
  * 假设$x_t$仅仅与其前$\tau$个变量有关
  * $x_t ~ P(x_t|x_{t-1}, ..., x_{t-\tau})$
* 隐变量自回归模型
  * 将$x_1, ... , x_{t-1}$总结为$h_t$
  * $\hat{x_t} = P(x_t|h_t)$来估计$x_t$
  * $h_t = g(h_{t-1}, x_{t-1})$
    ![序列模型](./sequence-model.svg)

## 文本预处理

* 词元化（tokenize）
  * 根据单词词元化
    * "Deep learning is fun" $\Rarr$ ['Deep', 'learning', 'is', 'fun']
  * 根据字符词元化
    * "I am a cat" $\Rarr$ ['I', ' ', 'a', 'm', ' ', 'c', 'a', 't']
* 词表（vocabulary）
  * 将词元(token)映射为数字的表
  * 唯一词元集合称为语料(corpus)

## 循环神经网络（隐变量自回归模型）

* 隐变量自回归模型
  * 将$x_1, ... , x_{t-1}$总结为$h_t$
  * $\hat{x_t} = P(x_t|h_t)$来估计$x_t$
  * $h_t = g(h_{t-1}, x_{t-1})$
    ![序列模型](./sequence-model.svg)
  * $\hat{x_t} = f(h_t) = h_t \cdot W + b$
  * $h_t = \phi(x_{t-1} \cdot W_1 + h_{t-1} \cdot W_2 + b)$
* 梯度剪裁
  * 防止梯度爆炸
  * 将参数的梯度$\vec g$的模限制在$\theta$内
  * $\vec g = min(1, {\theta \over |\vec g|}) \cdot \vec g$
  * $\theta$一般取1
