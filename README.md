# Operator Splitting, Policy Iteration, and Machine Learning for Stochastic Optimal Control

[![arXiv](https://img.shields.io/badge/arXiv-2603.12167-b31b1b.svg)](https://arxiv.org/abs/2603.12167)
[![Paper HTML](https://img.shields.io/badge/Read-paper%20HTML-2563eb.svg)](https://arxiv.org/html/2603.12167v1)
[![TensorFlow](https://img.shields.io/badge/Implementation-TensorFlow%20compat.v1-ff6f00.svg)](https://www.tensorflow.org/guide/migrate)

**A value–gradient policy-iteration framework for Hamilton–Jacobi–Bellman (HJB) equations, coupling characteristic-based machine learning with operator splitting for stochastic optimal control.**

This repository accompanies our paper:

> **Alain Bensoussan, Thien P. B. Nguyen, Minh-Binh Tran, and Son N. T. Tu (2026).**  
> **Operator Splitting, Policy Iteration, and Machine Learning for Stochastic Optimal Control.**  
> arXiv:2603.12167 · [Paper](https://arxiv.org/abs/2603.12167) · [Version 1 (HTML)](https://arxiv.org/html/2603.12167v1) · [PDF](https://arxiv.org/pdf/2603.12167)

**Author's implementation:** The repository is maintained by **Thien P. B. Nguyen**, a co-author of the paper. Its Python scripts illustrate the numerical methods and test regimes underlying **Tables 1, 2, and 4** of Section 4. The source files are research scripts, not a packaged exact-reproduction pipeline; important differences between the published experimental specification and current code are documented [below](#9-important-paper-vs-code-differences).

---

## Highlights

- **Split a difficult second-order PDE** into a nonlinear, first-order Hamilton–Jacobi problem and a linear heat equation.
- **Learn values and gradients together** along sampled characteristic trajectories, avoiding a full \(d\)-dimensional Cartesian grid.
- **Use PI-\(\lambda\)** (value–gradient policy iteration) to reduce policy evaluation to linear characteristic equations with an explicit optimization-based policy update.
- **Obtain rigorous convergence estimates:** regularity-dependent splitting error bounds, an \(L^1\) estimate in the periodic setting, and geometric decay of the ideal PI-\(\lambda\) gradient-iteration error.
- **Explore deterministic and stochastic benchmarks:** \(d=32\) without diffusion and \(d=5\) with two different diffusion strengths.

### Code-to-paper map

| Paper | \(\varepsilon\) | \(h\) | \(d\) | Python script | Interpretation |
|:--|--:|--:|--:|:--|:--|
| **Table 1** | 0 | 0.01* | 32 | [`horizon_HJB`](./horizon_HJB) | **No operator-splitting / heat step is needed** |
| **Table 2** | 0.01 | 0.1 | 5 | [`horizon_HJB_heat(2)`](./horizon_HJB_heat%282%29) | Low-diffusion heat/HJ experiment |
| **Table 4** | 1 | 0.05 | 5 | [`horizon_HJB_heat`](./horizon_HJB_heat) | Higher-diffusion heat/HJ experiment |

*The paper reports \(h=0.01\) for Table 1. In that script, `dt_hjb=0.01` is a numerical time increment; since \(\varepsilon=0\), no nontrivial heat operator is required.*

> **Scientific reproducibility:** These are the table-associated files, but the **checked-in versions do not implement all paper conventions verbatim**. In particular, test-matrix construction, the cost/initial data, control updates, splitting order, training settings, and the measured residual differ. Do not assume that executing the files as-is exactly recreates all numerical entries printed in the article.

## Contents

1. [The HJB problem](#1-the-hjb-problem)
2. [Operator splitting](#2-operator-splitting)
3. [PI-\(\lambda\): value–gradient policy iteration](#3-pi-λ-valuegradient-policy-iteration)
4. [Characteristic-based machine learning](#4-characteristic-based-machine-learning)
5. [Parameter glossary: \(\mu,N,T1,T2,\varepsilon,h,d\)](#5-parameter-glossary)
6. [Numerical experiments](#6-numerical-experiments)
7. [Theoretical guarantees](#7-theoretical-guarantees)
8. [How to run the code](#8-how-to-run-the-code)
9. [Important paper-vs-code differences](#9-important-paper-vs-code-differences)
10. [Citation](#10-citation)

---

## 1. The HJB problem

We study the viscous Hamilton–Jacobi equation

$$
\boxed{\partial_t u(x,t)+H(x,\nabla u(x,t))=\varepsilon\Delta u(x,t),
\qquad u(x,0)=u_0(x),}
$$

where \(x\in\mathbb R^d\), \(t\in[0,T]\), \(u(x,t)\) is a value function, \(H\) is a Hamiltonian, and \(\varepsilon\ge0\) controls diffusion. The **deterministic** case \(\varepsilon=0\) is first order; the **stochastic** case \(\varepsilon>0\) includes diffusion.

When the Hamiltonian comes from control dynamics \(f(x,a)\) and running cost \(\ell(x,a)\), the paper uses

$$
H(x,p)=\sup_a\{-p\cdot f(x,a)-\ell(x,a)\}
=-\inf_a\{p\cdot f(x,a)+\ell(x,a)\}.
$$

The deterministic value function is

$$
\begin{aligned}
u(x,t)&=\inf_{a(\cdot)}\left\{\int_0^t
\ell(\eta(s),a(s))\,ds+u_0(\eta(0))\right\},\\
\dot\eta(s)&=-f(\eta(s),a(s)),\qquad \eta(t)=x.
\end{aligned}
$$

The endpoint \(\eta(t)=x\) is fixed and trajectories are traced toward the initial time. A conventional grid with \(m\) nodes per coordinate needs \(m^d\) states. The characteristic approach instead constructs supervised data along sampled paths, reducing the dependence on a dense high-dimensional spatial grid (without removing the underlying dimension dependence of learning and sampling).

## 2. Operator splitting

Write \(\mathcal S_h^{\mathrm{HJ}}\) for the first-order evolution operator and \(\mathcal S_h^{\mathrm H}\) for the heat operator.

**Nonlinear Hamilton–Jacobi step:**

$$\partial_t w+H(x,\nabla w)=0,\qquad w(\cdot,0)=v.$$

This is the computationally challenging control part; the paper treats it with PI-\(\lambda\), characteristics, and machine learning.

**Linear heat step:**

$$\partial_t z=\varepsilon\Delta z,\qquad z(\cdot,0)=v.$$

The heat semigroup admits an explicit Gaussian representation:

$$
\begin{aligned}
(\mathcal S_h^{\mathrm H}v)(x)
&=\int_{\mathbb R^d}\frac{e^{-|x-y|^2/(4\varepsilon h)}}{(4\pi\varepsilon h)^{d/2}}v(y)\,dy\\
&=\mathbb E\big[v(x+\sqrt{2\varepsilon h}\,Z)\big],\quad
Z\sim\mathcal N(0,I_d).
\end{aligned}
$$

This motivates Monte Carlo heat updates. When \(\varepsilon=0\), the heat operator becomes the identity.

For \(h=T/n\) and \(t_i=ih\), **Equation (1.5) in the paper applies the full HJ step first, then the full heat step**:

$$
\boxed{v_i=\mathcal S_h^{\mathrm H}\circ\mathcal S_h^{\mathrm{HJ}}\,v_{i-1},
\qquad v_0=u_0.}
$$

```text
Value v at t_(i-1)
       │
       ▼
First-order HJ evolution over h
  └─ PI-lambda + characteristic learning
       │
       ▼
Heat evolution over h (only if epsilon > 0)
  └─ Gaussian convolution / numerical heat solver
       │
       ▼
Value v at t_i
```

This Lie/Trotter-type splitting separates nonlinearity from diffusion. **The checked-in heat scripts instead use a heat-half update followed by a first-order HJ-half update**; that computational composition is not identical to the full-step operator formula above. See Section 9 for a precise distinction.

## 3. PI-\(\lambda\): value–gradient policy iteration

Instead of directly solving the nonlinear first-order HJB equation, the method iterates over policies and the **value gradient**

$$\lambda(x,t)=\nabla_x u(x,t).$$

### Step 1 — linear policy evaluation

Given the current policy \(a^{(k)}(x,t)\), **Algorithm 1** computes the next value gradient by solving

$$
\boxed{\begin{aligned}
\partial_t\lambda^{(k+1)}
&=D\lambda^{(k+1)}\,f(x,a^{(k)})
+D_xf(x,a^{(k)})^\top\lambda^{(k)}
+\nabla_x\ell(x,a^{(k)}),\\
\lambda^{(k+1)}(x,0)&=\nabla u_0(x).
\end{aligned}}
$$

All terms involving \(a^{(k)}\) or \(\lambda^{(k)}\) are known, so this is **linear in the new iterate** \(\lambda^{(k+1)}\). The \(d\) gradient components can be evaluated along characteristic paths with parallelism.

For the characteristic \(X(s)\) defined by

$$\dot X(s)=-f(X(s),a^{(k)}(X(s),s)),$$

we obtain the ODE

$$
\frac{d}{ds}\lambda^{(k+1)}(X(s),s)
=D_xf(X(s),a^{(k)})^\top\lambda^{(k)}(X(s),s)
+\nabla_x\ell(X(s),a^{(k)}).
$$

This turns PDE evaluation into trajectory integrations and labeled gradient samples. The current scripts implement state paths in `make_trajectory_backward` and gradient labels in `integrate_lambda_along_trajectory` using a backward RK4-style update. The source-code gradient ODE and boundary labels are implementation choices and should not be mistaken for an exact symbolic implementation of every formula in Algorithm 1.

### Step 2 — policy improvement

Given \(\lambda^{(k+1)}\), the policy is improved pointwise via

$$
\boxed{a^{(k+1)}(x,t)=\arg\min_a\left\{
\lambda^{(k+1)}(x,t)\cdot f(x,a)+\ell(x,a)\right\}.}
$$

Equivalently, maximize \(-\lambda^{(k+1)}\cdot f-\ell\). Repeat the two steps until the gradient, policy, or residual meets a stopping criterion. **In the public scripts, `force_zero=True` fixes characteristic controls to zero by default; this turns off the full feedback-policy improvement in the executed runs**, even though an action-computation helper is present.

### Conceptual algorithm

```text
Input: initial value u0, dynamics f, cost ell, epsilon, horizon T, step h
Set v0 = u0

for each splitting interval:
    initialize a^(0) and lambda^(0)
    for k = 0, 1, ... (policy iterations):
        sample characteristic starting/ending states
        integrate fixed-policy state trajectories
        obtain supervised value and gradient labels
        train differentiable approximation V_theta
        compute updated lambda from grad_x V_theta
        improve a^(k+1) by pointwise minimization
    evolve the first-order HJ solution for h
    evolve the linear heat equation for h when epsilon > 0
    retain the new value function

Output: approximated value function and feedback policy (if activated)
```

The above is the **paper's mathematical workflow**. The current TensorFlow implementation contains approximations and fixed-control choices noted later.

## 4. Characteristic-based machine learning

### 4.1 Reduce time-discrete evaluation to a transport equation

Section 4.1 introduces, for a discrete time interval \(h\),

$$
\begin{aligned}
\mathcal V(x)&=u^{(k)}(x,s+h), &\mathcal G(x)&=f(x,a^{(k)}(x,s)),\\
\mathcal L(x)&=\ell(x,a^{(k)}(x,s))+h^{-1}u^{(k)}(x,s),
&\kappa&=h^{-1}.
\end{aligned}
$$

The linear time-discrete equation becomes the stationary transport problem

$$\boxed{\kappa\mathcal V(x)-\mathcal G(x)\cdot\nabla\mathcal V(x)=\mathcal L(x).}$$

Along \(\dot X(\tau)=\mathcal G(X(\tau))\), the paper derives the characteristic representation

$$
\mathcal V(X(\tau))=\int_\tau^\infty e^{-\kappa(r-\tau)}\mathcal L(X(r))\,dr,
$$

under an appropriate decay condition, along with an associated gradient representation (Equations **4.3–4.4**). Numerical trajectories and gradient/value labels therefore replace a dense spatial PDE grid.

### 4.2 Fit the value **and** its gradient

Let \(\widehat{\mathcal V}_\theta\) be a differentiable model (here, a tanh neural network). The **paper's loss, Equation (4.5)**, is

$$
\boxed{\begin{aligned}
\mathcal J_{\mathrm{paper}}(\theta)
={}&\mu\sum_{n=1}^{N}\int_0^T
\left|\mathcal V(X^{(n)}(\tau))-\widehat{\mathcal V}_{\theta}(X^{(n)}(\tau))\right|^2\,d\tau\\
&+(1-\mu)\sum_{n=1}^{N}\int_0^T
\left\|\lambda(X^{(n)}(\tau))-\nabla\widehat{\mathcal V}_{\theta}(X^{(n)}(\tau))\right\|^2\,d\tau.
\end{aligned}}
$$

- **Value matching** fits the scalar value labels derived along trajectories.
- **Gradient matching** fits the vector \(\lambda=\nabla u\), directly relevant to feedback control.
- **\(\mu\in[0,1]\)** weights the **value** term in the *paper*; \(1-\mu\) weights the **gradient** term. The limits \(\mu=1\) and \(\mu=0\) use value-only and gradient-only fitting, respectively.
- Automatic differentiation computes \(\nabla\widehat{\mathcal V}_\theta\). The paper uses ADAM and replaces the integrals by sums over discrete path snapshots, recommending approximately uniform **spatial/arc-length** coverage.

### 4.3 Extra loss used in the public code

The source scripts augment the two-term loss with a PDE penalty:

$$
\boxed{\mathcal J_{\mathrm{code}}
=\underbrace{\eta\,\mathcal J_{\mathrm{value}}}_{\text{value fit}}
+\underbrace{(1-\eta)\,\mathcal J_{\mathrm{gradient}}}_{\text{gradient fit}}
+\underbrace{\beta\,\mathcal J_{\mathrm{PDE}}}_{\text{extra penalty}}.}
$$

**Important naming difference:** The code sets `mu = 1 - eta` and uses `value_weight = eta`, `gradient_weight = mu`. Hence **the paper's \(\mu\) corresponds to the code's `eta`, not the variable called `mu`**.

| Value weight in paper \(\mu\) | Set code `eta` | Code `mu = 1 - eta` (gradient weight) |
|--:|--:|--:|
| 0.2 | 0.2 | 0.8 |
| 0.4 | 0.4 | 0.6 |
| 0.6 | 0.6 | 0.4 |
| 0.8 | 0.8 | 0.2 |

The additional weight `pde_weight` equals **10** in `horizon_HJB` and `horizon_HJB_heat`, and **20** in `horizon_HJB_heat(2)`. It is an extension of the paper's displayed Equation (4.5), not a term to silently assume is present in that equation.

## 5. Parameter glossary

| Symbol | Meaning | Notes for experiments and code |
|:--|:--|:--|
| **\(\mu\)** | Value–gradient loss balance | The **paper's \(\mu\)** weights the **value loss**, \(1-\mu\) the gradient loss. Test grid: \(0.2,0.4,0.6,0.8\). In code, set `eta` to the desired paper \(\mu\). |
| **\(N\)** | Number of **characteristic trajectories** | Test grid: \(12,14,16,18,20\). Code `tranum` / `tranum_all`. **Not** the number of snapshots per trajectory or splitting intervals. |
| **T1** | Experimental **Test 1**, choice of \(A\) | \(A=I_d\) in Section 4.2 of the paper. |
| **T2** | Experimental **Test 2**, choice of \(A\) | \(A=(\mathfrak a^\top\mathfrak a+I_d)/d\), \(\mathfrak a_{ij}\sim\mathcal N(0,1)\) i.i.d. in the paper. |
| **\(\varepsilon\)** | Viscosity/diffusion coefficient | \(0\) deterministic; \(>0\) stochastic. Heat increments have scale \(\sqrt{2\varepsilon h}\). Code `epsilon`. |
| **\(h\)** | Time-discretization / splitting-step length | Paper \(h=T/n\). In heat code, outer `dt=T_scheme/N_time`; separate inner `dt_hjb=0.01`. |
| **\(d\)** | State-space dimension | \(x\in\mathbb R^d\). Code `xdim`: **32** for Table 1, **5** for Tables 2 and 4. |
| \(\lambda\) | Value gradient \(\nabla_xu\) | Estimated by `lambda_approx` and auto-differentiated from the NN. |
| \(a\) | Control policy | Determines \(f(x,a)\); default scripts force it to zero. |
| \(\theta\) | NN parameters | Weights and biases used in supervised fitting. |
| \(n\) / `N_time` | Number of splitting intervals | Gives \(h=T/n\); **different from \(N\)**. |
| \(N_p\) | Number of test points for the residual | Paper: **10,000**; source evaluation loops: **1,000**. |

**Do not confuse experimental T1/T2 with theoretical \(T_1,T_2\).** Section 3 defines \(T_1,T_2\) as bounds on the time horizon used in the convergence proof. The **T1 and T2 labels in Tables 1–4 instead refer only to the two matrix benchmarks**.

**Interpretation:** Increasing \(N\) broadens possible path coverage but does not ensure monotonic error reduction; varying \(\mu\) changes the trade-off between function-value and derivative accuracy; increasing \(\varepsilon\) changes the underlying PDE; reducing \(h\) reduces ideal splitting error under the theorem's assumptions while increasing the number of outer steps; and increasing \(d\) challenges spatial sampling and learning capacity.

---

## 6. Numerical experiments

### 6.1 Quadratic control benchmark in Section 4.2

The paper considers the linear state dynamics and quadratic running cost

$$
f(x,a)=Ax+Ba,\qquad B=I_d,\qquad \ell(x,a)=|x|^2+|a|^2.
$$

Using the paper's sign convention for the Hamiltonian,

$$
\boxed{H(x,p)=\frac14|p|^2-p\cdot Ax-|x|^2},
$$

because the pointwise minimizer is \(a^*(x,p)=-p/2\) when \(B=I_d\). The paper states initial data \(u_0(x)=e^{-x^2}\) and a time interval \([0,1]\).

The **two experimental matrix cases** are

$$
\mathrm{T1}:\ A=I_d,\qquad
\mathrm{T2}:\ A=\frac{\mathfrak a^\top\mathfrak a+I_d}{d},
\quad\mathfrak a_{ij}\overset{\mathrm{i.i.d.}}\sim\mathcal N(0,1).
$$

**Paper's experimental design:** For each T1/T2 test, explore \(N=12,14,16,18,20\) trajectories and \(\mu=0.2,0.4,0.6,0.8\). Initial points are sampled from \([-1,1]^d\), and the paper describes 30 policy iterations, full-batch ADAM, up to 1,000 optimizer steps per policy iteration, and an average residual over the **last 20** policy iterations.

### 6.2 Table 1 — deterministic high-dimensional case

- **Paper parameters:** \(\varepsilon=0\), \(h=0.01\), \(d=32\).
- **Script:** [`horizon_HJB`](./horizon_HJB).
- **Main point:** Because diffusion vanishes, **no splitting/heat step is necessary**. The script focuses on characteristic learning with a `[64,64,64,1]` tanh value network, trajectory-derived targets, and iteration-wise residual plots.
- **Source defaults:** `xdim=32`, `dt_hjb=0.01`, `eta_all=[0.6]`, `tranum_all=[12,14,16,18,20]`, `itenum=50` and `force_zero_global=True`. Other choices of paper \(\mu\) require modifying `eta_all`.

### 6.3 Table 2 — lower-viscosity five-dimensional case

- **Paper parameters:** \(\varepsilon=0.01\), \(h=0.1\), \(d=5\).
- **Script:** [`horizon_HJB_heat(2)`](./horizon_HJB_heat%282%29).
- **Source defaults:** `epsilon=0.01`, `T_scheme=2.0`, `N_time=20`, `eta=0.6`, `tranum=12`, `itenum_pi=5`, `n_mc_heat=16`.
- **Step-size check:** `dt=T_scheme/N_time=0.1`, matching the paper's \(h\), but the code's **outer time horizon is 2**, not the paper's stated test interval \([0,1]\). The code applies Monte Carlo heat updates together with learned first-order updates.

### 6.4 Table 4 — higher-viscosity five-dimensional case

- **Paper parameters:** \(\varepsilon=1\), \(h=0.05\), \(d=5\).
- **Script:** [`horizon_HJB_heat`](./horizon_HJB_heat).
- **Source defaults:** `epsilon=1`, `T_scheme=1.0`, `N_time=20`, `eta=0.6`, `tranum=12`, `itenum_pi=5`, `n_mc_heat=16`.
- **Step-size check:** `dt=T_scheme/N_time=0.05`. Its `heat_half_step_mc` approximates Gaussian averaging by sampling perturbed locations and mapping each back to the closest working spatial point before averaging the stored values.

### 6.5 What is meant by “error” in the tables?

The **paper's reported error (Equation 4.7)** is the mean absolute residual of the **stationary, time-discrete transport equation**, sampled at \(N_p=10^4\) points:

$$
\boxed{\mathrm{error}
=\frac1{N_p}\sum_{j=1}^{N_p}\left|
\kappa\widehat{\mathcal V}_\theta(x^{(j)})
-\mathcal G(x^{(j)})\cdot\nabla\widehat{\mathcal V}_\theta(x^{(j)})
-\mathcal L(x^{(j)})\right|.}
$$

This is a **PDE residual**, not an error against a known exact solution. The paper evaluates T1 and T2 over the \((\mu,N)\) grid and reports averages over its final 20 PI iterations.

The scripts' printed `HJB Residual` is **not calculated with Equation (4.7)**. It instead uses approximately 1,000 randomly sampled time-space points to form

$$
\frac1{1000}\sum_j\left|
-\frac{\widehat V(x_j,t_j+\Delta t)-\widehat V(x_j,t_j)}{\Delta t}
-\left(\nabla_x\widehat V(x_j,t_j)\cdot Ax_j+\frac12|x_j|^2\right)
\right|,
\qquad \Delta t=\texttt{dt\_hjb},
$$

under a zero-control convention. The diagnostic does **not** explicitly include \(\varepsilon\Delta\widehat V\). Consequently, a source-code residual and a published table entry should not be compared as if they were the same statistic.

---

## 7. Theoretical guarantees

The following are statements about the **mathematical splitting and PI-\(\lambda\) procedures under the paper's assumptions**, not universal guarantees for finite neural networks, finite data, or the current source files.

### 7.1 One-sided splitting estimates (Theorem 1.1)

Let \(u\) be the exact viscous solution and \(v\) the splitting approximation. At splitting times \(t_i\), the paper proves

$$
\boxed{-CT\varepsilon h\le u(x,t_i)-v(x,t_i)}
$$

and the following **upper** bounds on the signed difference \(u-v\):

| Regularity of \(u_0\) | Upper bound on \(u(x,t_i)-v(x,t_i)\) |
|:--|:--|
| Lipschitz | \(C(T\varepsilon h)^{1/7}\) |
| Lipschitz and semiconcave | \(C(T\varepsilon h)^{1/5}\) |
| \(C^2\), with the stated boundedness assumptions | \(C(T\varepsilon h)^{1/3}\) |

These are **directional (one-sided)** inequalities, not claims that \(u-v\) always has one sign. The proof uses heat/HJ commutator estimates, Lipschitz and semiconcavity controls, and an auxiliary vanishing-viscosity regularization to obtain the upper estimates.

### 7.2 Periodic-domain bound (Proposition 1.2)

For periodic Hamiltonians and suitable semiconcave initial data,

$$
\boxed{\|u(\cdot,t_i)-v(\cdot,t_i)\|_{L^1(\mathbb T^d)}
\le C(e^{CT}-1)\sqrt{\varepsilon h}.}
$$

The conditions matter: this is not a universal \(L^1\) estimate on arbitrary domains without additional assumptions.

### 7.3 Geometric PI-\(\lambda\) convergence (Theorem 1.4)

For the weighted squared \(L^2\) difference between successive gradient iterates,

$$
e_k=\int_0^T e^{-\gamma t}\int_{\mathbb R^d}
\frac{|\lambda^{(k)}(x,t)-\lambda^{(k-1)}(x,t)|^2}
{(1+|x|^2)^{2\alpha}}\,dx\,dt,
$$

under the paper's structural assumptions, a sufficiently short time horizon, and an appropriate weight parameter \(\gamma\), the analysis yields

$$\boxed{e_k\le 2^{-(k-1)}e_1.}$$

In other words, the **ideal gradient-iteration discrepancy decays geometrically**. This theorem does not by itself establish the same iteration-by-iteration rate for an imperfectly trained neural network or for Monte Carlo discretization error.

---

## 8. How to run the code

### Repository layout

```text
Operator-Splitting-Policy-Iteration/
├── horizon_HJB             # Table 1-related, epsilon = 0
├── horizon_HJB_heat(2)     # Table 2-related, epsilon = 0.01
├── horizon_HJB_heat        # Table 4-related, epsilon = 1
└── README.md
```

These repository files contain Python code but **have no `.py` extension** in their current names.

### Dependencies

The scripts import **NumPy**, **Matplotlib**, and **TensorFlow** using `tensorflow.compat.v1`, as well as standard Python modules (`os`, `pickle`, `random`, `math`, etc.). An example installation is

```bash
python -m pip install numpy matplotlib tensorflow
```

The code calls `tf.disable_v2_behavior()` and uses placeholders, sessions, `tf.gradients`, and graph-mode optimizers. It therefore requires a TensorFlow installation exposing compatible **TF1-style APIs**. The repository does not provide pinned dependencies; compatibility with every current Python/TensorFlow version has **not** been verified here.

### Execute a script

```bash
git clone https://github.com/thienPhuoc624/Operator-Splitting-Policy-Iteration.git
cd Operator-Splitting-Policy-Iteration

# Deterministic case / Table 1-related
python horizon_HJB

# Low diffusion / Table 2-related
python 'horizon_HJB_heat(2)'

# Higher diffusion / Table 4-related
python horizon_HJB_heat
```

**Edit the output path before running.** Each script defines an `address` under `/content/drive/MyDrive/...`, intended for a Colab/Google Drive-style environment. Either mount that location in Colab or change `address` to an existing writable local location. The scripts create directories and cache randomly generated data and test matrices as pickle files including `zinitial_r1.data`, `z_generated_A_r1.data`, and `z_errcheck_r1.data`. **Existing caches take precedence over freshly generated test data**, so check the saved \(A\) matrix when changing from T1 to T2.

### Important parameters

| Python name | Purpose | Default examples |
|:--|:--|:--|
| `xdim` | Dimension \(d\) | `32` (Table 1), `5` (Tables 2/4) |
| `epsilon` | Diffusion \(\varepsilon\) | `0.01` or `1` in heat scripts |
| `T_scheme`, `N_time` | Outer splitting horizon/number of intervals | `dt = T_scheme / N_time` |
| `dt_hjb` | Inner numerical time increment (distinct from outer `dt`) | `0.01` |
| `eta` | Value loss weight; equals paper's \(\mu\) | `0.6` |
| `mu = 1 - eta` | Gradient loss weight **in code** | `0.4` when `eta=0.6` |
| `tranum`, `tranum_all` | Number of paths \(N\) | `12` / `[12,14,16,18,20]` |
| `itenum`, `itenum_pi` | Loop count for PI evaluations | `50` in Table 1 file; `5` per outer step in heat files |
| `n_mc_heat` | Gaussian samples per heat evaluation point | `16` |
| `layers` | MLP architecture | `[64,64,64,1]` / `[128,128,64,1]` |
| `force_zero`, `force_zero_global` | Disable feedback control along paths | Set to `True` in source runs |

To explore a full grid, vary `eta` over `[0.2,0.4,0.6,0.8]` and `tranum` over `[12,14,16,18,20]`, separately for T1 and T2. **This is not sufficient by itself for exact replication**: the paper-vs-code differences in Section 9 must also be addressed.

### Main functions

| Function / component | Responsibility |
|:--|:--|
| `make_trajectory_backward` | Construct characteristic-related state paths under the selected (currently fixed-zero) control |
| `integrate_lambda_along_trajectory` | Produce gradient labels via RK4-style backward integration |
| `neural_net` and `lambda_approx` | Approximate the value function with an MLP and differentiate it with respect to state |
| `value_loss`, `gradient_loss`, `pde_loss` | Fit scalar values, value gradients, and an additional PDE consistency penalty |
| `pi_lambda_update_and_residual` | Train and evaluate the policy-iteration subproblem in heat scripts |
| `heat_half_step_mc` | Monte Carlo heat averaging with nearest-neighbor value lookup |
| `hj_half_step_using_nn` | Explicit first-order HJ field update using neural-network gradients |
| `run_splitting_scheme_with_pi_lambda` | Coordinate the outer heat/HJ time-loop in heat scripts |

In `horizon_HJB`, the main policy-iteration training and evaluation loop is written directly at top level rather than wrapped into `pi_lambda_update_and_residual`.

---

## 9. Important paper-vs-code differences

The paper defines the **method**, while the current scripts are **experimental implementations with additional assumptions and modifications**. For transparent reproduction, distinguish these carefully:

| Feature | Paper | Checked-in scripts |
|:--|:--|:--|
| Running cost | \(\ell(x,a)=|x|^2+|a|^2\) | `func2` / `running_cost` use \(\frac12(|x|^2+|a|^2)\) |
| Value boundary/initial data | \(u_0(x)=e^{-x^2}\) in Section 4.2 | `terminal_cost = 0.5*sum(x**2)` and its gradient are used as training labels |
| **T1 matrix** | \(A=I_d\) | Code initializes \(A_{\rm temp}=I_d\) and **shifts the eigenvalues**, giving \(-0.5I_d\) in that branch |
| **T2 matrix** | \((\mathfrak a^\top\mathfrak a+I_d)/d\), standard-normal entries | Uses small Gaussian entries scaled by `0.05`, `a @ a.T`, and an eigenvalue-stabilizing shift |
| Meaning of \(\mu\) | Value loss coefficient | **Code variable** `mu=1-eta` is gradient coefficient; `eta` is value coefficient |
| Training loss | Equation (4.5): value + gradient | Value + gradient **plus a separate PDE penalty** (`pde_weight=10` or `20`) |
| Training batches | Full-batch ADAM | Mini-batches of size `64` with learning-rate decay and early stopping |
| Iteration count | `30` iterations, average final `20` | Table 1 script uses `50`; heat scripts use `5` per splitting interval by default |
| Parameter sweeps | \(N=12,14,16,18,20\), \(\mu=0.2,0.4,0.6,0.8\) | Table 1 script sweeps \(N\) but fixes `eta=0.6`; heat scripts run one pair `eta=0.6`, `tranum=12` |
| Spatial samples | \([-1,1]^d\) | Generates values in \([-0.5,0.5]^d\), often multiplying by another `0.3` |
| Residual | Stationary Equation (4.7), 10,000 test points | Time-dependent, zero-control residual from 1,000 sampled points; no explicit viscous Laplacian |
| Splitting | Full first-order HJ step, **then** full heat step | Heat **half step**, then NN-based HJ **half step**; these are not the paper's exact operator composition |
| Updated splitting data | The HJ solve consumes the preceding splitting value \(v_{i-1}\) | The PI training stage does not directly condition its supervised targets on the newly obtained `u_prev` |
| Feedback | Policy improvement is the core of PI-\(\lambda\) | `force_zero=True` and zero `u_opt_batch` prevent an active feedback update in the current runs |
| Table 2 horizon | Experiment described on \([0,1]\) | `T_scheme=2.0`, although `dt=0.1` matches the listed \(h\) |

The running-cost normalization changes the optimal policy: for the **paper's** \(\ell=|x|^2+|a|^2\) and \(B=I_d\), \(a^*=-p/2\); for the **script's** \(\ell=\tfrac12(|x|^2+|a|^2)\), the corresponding unconstrained minimizer is \(a^*=-p\). However, the provided runs currently **force \(a=0\)**.

**Reproduction guidance:** To claim numerical agreement with any published table, align the test matrices, initial data, cost normalization, control update, splitting composition and step sizes, learning configuration, residual definition, and random seeds **before** comparing numbers. Running the public scripts unchanged is better described as **exploring related experimental configurations**, not a verified bit-for-bit reproduction of the reported results.

### Reproducibility checklist

- [ ] Choose the matching script and \((d,\varepsilon,h)\) from the table map.
- [ ] Choose **T1** or **T2**, and verify any cached \(A\) matrix.
- [ ] Set the paper \(\mu\) via **code `eta`**, and trajectory count \(N\) via `tranum`.
- [ ] Decide whether to study the current **fixed-zero-control** version or implement active policy improvement.
- [ ] Match the cost normalization, initial data, and time horizon to the paper.
- [ ] Match the splitting order and full-step vs half-step composition.
- [ ] Use the **same residual formula, sample size, and averaging rule** as the paper.
- [ ] Record the environment, random seeds, network architecture, training steps, and cache state.

---

## 10. Citation

If you use this repository or the method in your research, please cite our paper:

```bibtex
@misc{bensoussan2026operatorsplitting,
  title         = {Operator Splitting, Policy Iteration, and Machine Learning for Stochastic Optimal Control},
  author        = {Bensoussan, Alain and Nguyen, Thien P. B. and Tran, Minh-Binh and Tu, Son N. T.},
  year          = {2026},
  eprint        = {2603.12167},
  archivePrefix = {arXiv},
  primaryClass  = {math.OC},
  url           = {https://arxiv.org/abs/2603.12167}
}
```

**Paper:** https://arxiv.org/abs/2603.12167  
**Code:** https://github.com/thienPhuoc624/Operator-Splitting-Policy-Iteration

*For the complete mathematical assumptions, formal proofs, and published numerical tables, consult the paper directly.*