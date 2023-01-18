# RNN

## 应用场景

* 时间序列预测
  * 2, 4, 6, 8, 10 ... 预测偶数
  * 股市波动预测
  * 物体温度曲线预测
* 股票价格预测（富时100指数）
    ![FTSE 100 Index](./ftse100.png)
    $x_t ~ P(x_t|x_{t-1}, ..., x_{1})$
    
    即 $x_t$ 与 $x_{t-1}, ..., x_{1}$ 有关

## 训练方法

* 自回归模型
  * 假设 $x_t$ 仅仅与其前 $\tau$ 个变量有关
  * $x_t ~ P(x_t|x_{t-1}, ..., x_{t-\tau})$
* 隐变量自回归模型
  * 将 $x_1, ... , x_{t-1}$ 总结为 $h_t$
  * $\hat{x_t} = P(x_t|h_t)$ 来估计 $x_t$
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
  * 将 $x_1, ... , x_{t-1}$ 总结为 $h_t$
  * $\hat{x_t} = P(x_t|h_t)$ 来估计 $x_t$
  * $h_t = g(h_{t-1}, x_{t-1})$
    ![序列模型](./sequence-model.svg)
  * $\hat{x_t} = f(h_t) = h_t \cdot W + b$
  * $h_t = \phi(x_{t-1} \cdot W_1 + h_{t-1} \cdot W_2 + b)$
* 梯度剪裁
  * 防止梯度爆炸
  * 将参数的梯度 $\vec g$ 的模限制在 $\theta$ 内
  * $\vec g = min(1, {\theta \over |\vec g|}) \cdot \vec g$
  * $\theta$ 一般取1


## 编码器解码器（Encoder、Decoder）

* 编码器的输出作为解码器的输入或者输入的一部分

![EncoderDecoder](/RNN/encoder-decoder.svg)

* Encoder

```#@save
class Seq2SeqEncoder(d2l.Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state
```

* Decoder

```
class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state
```

* EncoderDecoder

```
#@save
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

### 注意力机制

* 建模：给定查询 $\vec q$ 以及键值对 $(\vec k_i, \vec v_i),i \epsilon [1, n]$
* $f(\vec q) = \sum \alpha(\vec q, \vec k_i) \cdot \vec v_i$
* $\sum \alpha(\vec q, \vec k_i) = 1$ 称作注意力权重
* $\alpha(\vec q, \vec k_i) = softmax(a(\vec q, \vec k_i))$ ， $a(\vec q, \vec k_i)$ 称作注意力评分函数
* 加性注意力（additive attention）
  * $a(\vec q, \vec k_i) = W_v tanh(W_q \cdot \vec q + W_k \cdot \vec k)$
* 放缩点积注意力
  * $a(\vec q, \vec k_i) = \frac{{\vec q}^T \cdot \vec k_i}{\sqrt d}$
  * $d = dimension(\vec q)$