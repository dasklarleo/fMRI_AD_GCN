# Attention Mechanism

## what is attention？

Attention 就是计算当前的一个值 **(query)** 和其余的值 **（keys）** 的相关性（可以有很多方法实现例如直接点乘、矩阵乘等方法，）。求出了一个相关性的值之后，再经过一个**softmax**进行一个归一化的处理，处理完之后我们便得到了query同每一个key的一个相关性。再将这个相关性和**value** （一般key=value）点乘，便有了每个key对当前query的影响程度
## 