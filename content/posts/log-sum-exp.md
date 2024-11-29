---
title: "LogSumExp and block-wise softmax"
date: 2024-11-28T23:41:01+08:00
mathjax: true
---

Consider the softmax function
$$
softmax(x_i, [x_j]) = \frac{e^{x_i}}{\sum^{N}_{j=0} e^{x_j}}
$$
which is numerically unstable if x is too large.

Given that $x_i = x_i - max([x_j]) + max([x_j])$, we have
{{< texd `\begin{aligned}
    e^{x_i} &= e^{x_i - max([x_j]) + max([x_j])} \\
        &= e^{x_i - max([x_j])}e^{max([x_j])} \\
\end{aligned}` >}}

As a result, we have
{{< texd `\begin{aligned}
    softmax(x_i, [x_j]) &= \frac{e^{x_i - max([x_j])}e^{max([x_j])}}{\sum^{N}_{j=0} e^{x_j - max([x_j])}e^{max([x_j])}} \\
        &= \frac{e^{x_i - max([x_j])}}{\sum^{N}_{j=0} e^{x_j - max([x_j])}}
\end{aligned}` >}}
where $e^{x_i - max([x_j])}$ is numerically stabler than $e^{x_i}$.

Furthermore, we can save the sum of expotential in the log space, i.e.

{{< texd `
\begin{aligned}
LSE([x_j]) &= \log{\sum^{N}_{j=0} e^{x_j}} \\
    &= max([x_j]) + \log{\sum e^{x_j - max([x_j])}}
\end{aligned}
`>}}

In python, we can have
```python
def LSE_softmax(x):
    """
    x = [x_j] is a vector
    returns softmax result and log sum exp
    """
    m = x.max()
    a = (x - m).exp()
    b = a.sum()
    lse = m + torch.log(b)
    return a / b, lse
```


Given a new sequence $[x_k]$, we can calculate the sum of exponentials of two sequences by
$$
\sum e^{x_j} + \sum e^{x_k} = \exp{LSE([x_j])} + \exp{LSE([x_k])}
$$
Then, we can calculate the new LSE by
{{< texd `
\begin{aligned}
LSE([x_j] || [x_k]) &= \log{(\sum e^{x_j} + \sum e^{x_k})} \\
                    &= \log{(\exp{LSE([x_j])}(1 + \exp{LSE([x_k])} / \exp{LSE([x_j])}))} \\
                    &= LSE([x_j]) + \log{(1 + \exp{LSE([x_k])} / \exp{LSE([x_j])})} \\
                    &= LSE([x_j]) + \log{(1 + \exp{(LSE([x_k]) - LSE([x_j]))})}\\
\end{aligned}  \tag{1} \label{eq:sum_lse}
`>}}

As a result, we can calcuate the softmax of the concatention of two sequences by

{{< texd `
\begin{aligned}
softmax(x_i, [x_j] || [x_k]) &= \frac{e^{x_i}}{\sum e^{x_j} + \sum e^{x_k}} \\
                            &= \frac{softmax(x_i, [x_j]) * \sum e^{x_j}}{\sum e^{x_j} + \sum e^{x_k}} \\
                            &= \frac{softmax(x_i, [x_j]) * \exp{LSE([x_j])}}{\exp{LSE([x_j])} + \exp{LSE([x_k])}} \\
                            &= \frac{softmax(x_i, [x_j])}{1 + \exp{LSE([x_k])} / \exp{LSE([x_j])}} \\
                            &= \frac{softmax(x_i, [x_j])}{1 + \exp{(LSE([x_k]) - LSE([x_j]))}} \\
\end{aligned} \tag{2} \label{eq:sum_softmax}
`>}}

By equations \ref{eq:sum_lse} and \ref{eq:sum_softmax}, we have the following in Python
```python
def concat_softmax_and_lse(
    s1: torch.Tensor,
    lse1: torch.Tensor,
    s2: torch.Tensor,
    lse2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    new_lse = lse1 + torch.log(1 + torch.exp(lse2 - lse1))
    # new_s1 = s1 / (1 + torch.exp(lse2 - lse1))
    # new_s1 = s1 / exp(new_lse - lse1)
    # new_s1 = s1 * exp(lse1 - new_lse)
    new_s1 = s1 * torch.exp(lse1 - new_lse)
    # new_lse == lse2 / (1 + torch.exp(lse1 - lse2))
    new_s2 = s2 * torch.exp(lse2 - new_lse)
    output = torch.cat([new_s1, new_s2])
    return output, new_lse
```

