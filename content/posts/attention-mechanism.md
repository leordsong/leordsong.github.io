---
title: "Backprop for Attention Mechanism"
date: 2024-12-10T13:47:15+08:00
mathjax: true
---

Consider the attention function

$$
attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_{qk}}})V
$$

where $Q = X_q W_q$, $K = X_k W_k$ and $V = X_v W_v$ with shapes $X_q \in \mathbb{R}^{lq \times d_{qk}}$,
$X_k \in \mathbb{R}^{lkv \times d_{qk}}$, $X_v \in \mathbb{R}^{lkv \times d_{v}}$.
In practive, $d_{qk} = d_v$ and $lkv \geq lq$ with kv cache.

Now, given that $A = softmax(\frac{QK^T}{\sqrt{d_{qk}}})$, we can calculate the gradients w.r.t $W_v$:
{{< texd`
\begin{equation}
\begin{aligned}
    \frac{\partial L}{\partial W_v} &= \frac{\partial L}{\partial O} \frac{\partial O}{\partial V} \frac{\partial V}{\partial W_v} \\ &= \frac{\partial L}{\partial O} A x_v
\end{aligned}
\end{equation}
`>}}
Note that the above equation does not guarantee the dimension is aligned.


In python, we can have
```python
l_q, l_kv, d_qk, d_v = 5, 7, 8, 4

Wq = nn.Parameter(torch.rand(d_qk, d_qk))
Wk = nn.Parameter(torch.rand(d_qk, d_qk))
Wv = nn.Parameter(torch.rand(d_v, d_v))
xq = torch.rand(l_q, d_qk)
xk = torch.rand(l_kv, d_qk)
xv = torch.rand(l_kv, d_v)
Q = xq @ Wq
K = xk @ Wk
V = xv @ Wv

# A with shape (seq_len_q, seq_len_kv)
A = F.softmax((Q @ K.T) / math.sqrt(d_qk), dim=1)
O = A @ V # O with shape (seq_len_q, dv)
L = O.sum() # L is positive for gradient descent
L.backward()

# \frac{\partial L}{\partial O}
dL_dO = torch.ones(d_v, l_q)
# sum = (O x 1[dv, 1]).T x 1[seq_len_q, 1]

# \frac{\partial L}{\partial Wv}
dL_dWv = (dL_dO @ A @ xv).T
print(dL_dWv, Wv.grad)
```

Invoving softmax is a little bit complicated.
The Jacobian matrix $J$ for the softmax function is a square $ n \times n $ matrix, $\mathbf{z} = softmax(\mathbf{x})$ and $\mathbf{x}, \mathbf{z} \in \mathbb{R}^n$, and can be written as:

{{< texd`
\begin{equation} \label{eq:softmax}
    J = \begin{pmatrix}
    z_1(1 - z_1) & -z_1 z_2 & \cdots & -z_1 z_n \\
    -z_2 z_1 & z_2(1 - z_2) & \cdots & -z_2 z_n \\
    \vdots & \vdots & \ddots & \vdots \\
    -z_n z_1 & -z_n z_2 & \cdots & z_n(1 - z_n)
\end{pmatrix}
\end{equation}
`>}}

As a result, for $X \in \mathbb{R}^{m \times n}$, we have $J \in \mathbb{R}^{m \times n \times n}$.
In python, we can have
```python
def softmax_backprop(dL_dA, A):
    # diag_A - AA^T
    diag_A = torch.diag_embed(A)
    J = diag_A - \
        torch.bmm(A.unsqueeze(-1), A.unsqueeze(-2))
    # Compute dL_dQK
    return torch.bmm(dL_dA.unsqueeze(1), J).squeeze(1)
```

The gradients w.r.t $W_q$:
{{< texd`
\begin{equation}
\begin{aligned}
    \frac{\partial L}{\partial W_q} &= \frac{\partial L}{\partial O} \frac{\partial O}{\partial A} \frac{\partial A}{\partial QK} \frac{\partial QK}{\partial Q} \frac{\partial Q}{\partial W_q} \\ &= \frac{\partial L}{\partial O} V J \frac{K^T}{\sqrt{d_{qk}}} x_q
\end{aligned}
\end{equation}
`>}}

In python, we can have
```python
dL_dQK = softmax_backprop(dL_dO.T @ V.T, A)
dL_dWq = ((dL_dQK @ K / math.sqrt(d_qk)).T @ xq).T
print(dL_dWq, Wq.grad)
```

The gradients w.r.t $W_k$:
{{< texd`
\begin{equation}
\begin{aligned}
    \frac{\partial L}{\partial W_q} &= \frac{\partial L}{\partial O} \frac{\partial O}{\partial A} \frac{\partial A}{\partial QK} \frac{\partial QK}{\partial K} \frac{\partial K}{\partial W_k} \\ &= (\frac{\partial L}{\partial O} V J)^T \frac{Q}{\sqrt{d_{qk}}} x_k
\end{aligned}
\end{equation}
`>}}

In python, we can have
```python
dL_dWk = ((dL_dQK.T @ Q / math.sqrt(d_qk)).T @ xk).T
print(dL_dWk, Wk.grad)
```