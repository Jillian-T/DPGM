# DPGM: Dynamic Poisson Gamma Membership Model
## Introduction
DGPM is a probabilistic model, it can describe latent node-group memberships after observing the interactions between nodes.
I implemented the gamma process edge partition model for static networks, and hierarchical gamma process edge partition models for large sparse dynamic networks

## Requirements
pytorch>=1.8

## Code
### 2.1 Data
The synthetic data is a (6, 60, 60) tensor, it records the interactions between 60 nodes at 6 time snapshots.

### 2.2 Model
The model is expressed as:
```math
\begin{array}{l}
\phi^{(t)}_{nk} \sim {\rm Gam}(\phi^{(t-1)}_{nk}/\tau,1/\tau), \quad\textup{for \ t=1,...,T}
\\
\phi^{(0)}_{nk} \sim {\rm Gam}(g_0,1/h_0)
\\
r_k \sim {\rm Gam}(\gamma_0/K, 1/c_0)
\\
\lambda_{kk'} \sim \left\{ \begin{array}{ll} {\rm Gam}(\xi r_k, 1/\beta), & \textup{if } k=k' \\ {\rm Gam}(r_k r_{k'}, 1/\beta), & \textup{otherwise} \end{array}\right.
\\
x^{(t)}_{mn} \sim {\rm Po}(\displaystyle\sum_{k=1}^{K}\sum_{k'=1}^{K}\lambda_{kk'}\phi^{(t)}_{nk}\phi^{(t)}_{mk'})
\\
b^{(t)}_{mn} = \mathbb{1}(x^{(t)}_{mn}\geqslant1)
\end{array}
```

$\beta$, $c_0$, $g_0$, $h_0$ and $\tau$ are hyperparameters, others are model parameters

$\phi^{(t)}_{nk}$ : a positive value, measuring how strongly node $n$ is affiliated with community $k$ at time $t$.

$\phi^{(0)}_{nk}$ is the original state

$r_k$ : a positive value, indicates the popularity of community $k$

$\lambda_{kk'}$ : a positive value, measuring how strongly communities $k$ and $k'$ interact with each other

$b^{(t)}_{mn} = \mathbb{1}$ : there is interaction between node $m$ and node $n$ at time $t$

### 2.3 Inference
We present a gibbs sampling procedure to update hyperparameters and model parameters
