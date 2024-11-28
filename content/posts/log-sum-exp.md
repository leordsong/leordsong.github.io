---
title: "Log Sum Exp"
date: 2024-11-28T23:41:01+08:00
mathjax: true
---

Consider the softmax function
$$
softmax(x_i, [x_j]) = \frac{e^{x_i}}{\sum^{N}_{j=0} e^{x_j}}
$$
which is numerically unstable if x is too large.

Given that $ x_i - max([x_j]) + max([x_j]) = x_i$, we have
{{< texd `\begin{aligned}
    e^{x_i} &= e^{x_i - max([x_j]) + max([x_j])} \\
        &= e^{x_i - max([x_j]) + max([x_j])} \\
        &= e^{x_i - max([x_j])}e^{max([x_j])} \\
\end{aligned}` >}}

As a result, we have
{{< texd `\begin{aligned}
    softmax(x_i, [x_j]) &= \frac{e^{x_i - max([x_j])}e^{max([x_j])}}{\sum^{N}_{j=0} e^{x_j - max([x_j])}e^{max([x_j])}} \\
        &= \frac{e^{x_i - max([x_j])}}{\sum^{N}_{j=0} e^{x_j - max([x_j])}}
\end{aligned}` >}}
where $e^{x_i - max([x_j])}$ is numerically stabler than $e^{x_i}$.

Furthermore, we can save the sum of expotential in the log space, i.e.

$$
    \log{\sum^{N}_{j=0} e^{x_j}} = max([x_j]) + \log{\sum e^{x_j - max([x_j])}}
$$

Given a new sequence $\log{\sum^{N}_{k=0} e^{x_k}}$, we cant concat the two sequences by

$$
    \log{(\sum e^{x_j} + \sum e^{x_k})} = max([x_j], [x_k]) + \log{\sum e^{x_j - max([x_j])}} + max([x_j]) - max([x_k]) + \log{\sum e^{x_k - max([x_k])}}
$$

as 

$$
\log{\sum e^{x_j - max([x_j])}} + max([x_j]) - max([x_k]) = \log{(e^{max([x_j])}\sum e^{x_j - max([x_j])} / e^{max([x_k])})} = \log{\sum e^{x_j - max([x_k])}}
$$