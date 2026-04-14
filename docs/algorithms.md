# Algorithm Notes

## Problem Setup

The central target is a positive conditional ratio:

```text
r(x) ≈ E[y_num | x] / E[y_den | x]
```

In the campaign examples:

```text
y_num = spend
y_den = conversions
r(x) = expected cost per conversion
```

The main learner models the ratio directly instead of fitting spend and conversions separately and dividing the two predictions.

## Direct Ratio Modeling

Pick a convex scalar function `f` and define the predicted ratio as:

```text
r̂(x) = f′(⟨w, x⟩)
```

The per-example loss is:

```text
ℓ(w; x, y_num, y_den) = y_den · f(⟨w, x⟩) − y_num · ⟨w, x⟩ + (α/2)‖w‖²
```

If `z = ⟨w, x⟩`, then:

```text
∂ℓ/∂z = y_den · f′(z) − y_num
```

Taking conditional expectation given `x` shows why this targets the ratio:

```text
E[∂ℓ/∂z | x] = E[y_den | x] · f′(z) − E[y_num | x]
```

At the optimum the expected derivative is zero, so:

```text
f′(z) = E[y_num | x] / E[y_den | x]
```

That is the core idea behind the direct ratio learner.

## Link Functions

The repo uses positive link functions for the ratio:

- exponential: `r̂ = eᶻ`
- positive-part linear: `r̂ = max(0, z)`
- softplus: `r̂ = log(1 + eᶻ)`

These links keep the predicted ratio nonnegative and give different smoothness and tail behavior.

## Proximal Updates

The learner uses a proximal point update instead of a plain gradient step:

```text
wₜ₊₁ = argmin_u { ℓₜ(u) + (1/(2η))‖u − wₜ‖² }
```

This keeps each online update close to the previous iterate while still fitting the current observation.

The loss depends on `w` through `⟨w, x⟩`, which lets the vector-valued update reduce to scalar proximal operators. Some of those prox operators are closed form, while the softplus-integral case is solved with a short Newton iteration.

## Autoregressive Features

The feature builders keep rolling numerator and denominator histories and turn them into lagged ratio features.

The pattern is:

1. compute rolling means for numerator and denominator
2. map the local ratio onto a latent scale with a normalizer
3. keep a history of those normalized values as the feature vector

The normalizers match the geometry of the model. For example, the log-ratio normalizer matches the exponential link, and the inverse-softplus normalizer maps a positive ratio back to the latent scale of the softplus model.

In the experiment layer, the maintained panel builders use causal lag windows: the
feature vector for row `t` is built from previous observations only, never from the
current row's observed spend or count.

The experiment-layer synthetic panels use a simpler bounded periodic generator than the
stable library simulator, but they follow the same observation pattern: bounded latent
spend means and bounded latent true-ratio paths are sampled first, then observed count
is drawn from a Poisson model and observed spend from an overdispersed negative-binomial
model. In those experiment frames, `true_ratio` is the latent ratio of means rather than
the realized row-wise `spend / count`.

## Simulation Model

The simulator builds two smooth latent campaign processes from:

- a global trend
- a periodic trend
- a positive transform

Those latent signals are sampled independently for count and spend:

```text
count_trend = count_shift + count_latent
spend_trend = spend_scale · spend_latent²
```

Observed counts are drawn from a Poisson model. Observed spend is drawn from an overdispersed
negative binomial model with:

```text
Var[spend | μ] = 1 + dispersion · μ²
```

The ground-truth ratio is therefore known:

```text
true_ratio = spend_trend / count_trend
```

That makes it easy to compare online estimators against a known target.

## Baselines

The experiment layer includes simpler comparison models:

- linear ratio regression
- linear inverse-ratio regression
- ratio-of-regressors baselines
- exponential and quadratic online baselines
- simple decay-based estimators

These are useful comparison points, but the core library is organized around the direct ratio loss and its proximal updates.

The maintained benchmark report keeps the weighted mean log-error summary table and adds
weighted REC curves for the `tune`, `same`, and `shifted` splits. Those REC curves also
include the online `campaign_running_ratio` baseline, which predicts each campaign's
cumulative observed spend-to-count ratio before the current update.

## Further Reading In Code

The main implementation lives in:

- `src/ratio_estimation/models.py`
- `src/ratio_estimation/proximal.py`
- `src/ratio_estimation/features.py`
- `src/ratio_estimation/simulation.py`
