
## Bayesian GARCH(1,1) with PyMC

### Overview

This project implements a **Bayesian GARCH(1,1)** (Generalized Autoregressive Conditional Heteroskedasticity) model for estimating and forecasting financial return volatility. The model is implemented in **PyMC** and uses **Markov Chain Monte Carlo (MCMC)** to infer posterior distributions over volatility dynamics and tail behavior.

The notebook demonstrates a complete workflow from synthetic data generation to probabilistic risk metrics such as **Value-at-Risk (VaR)** and **Expected Shortfall (ES)**.

### Dependencies

Required Python libraries:

- `pymc`
- `arviz`
- `numpy`
- `matplotlib`
- `pytensor`

Install dependencies with:

```bash
pip install pymc arviz numpy matplotlib pytensor
````
### Workflow

### 1. Data Preparation

* Synthetic financial returns of length $T = 1000$ are generated.
* Returns follow a **Student’s t-distribution** to reflect heavy tails.
* The return series is demeaned before modeling.

### 2. Model Definition

A Bayesian **GARCH(1,1)** model is specified as follows.

#### Volatility Dynamics

The conditional variance evolves according to:

$$
\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2
$$

To enforce stationarity ($\alpha + \beta < 1$), the model uses the reparameterization:

$$
\alpha = \kappa \phi
$$

$$
\beta = \kappa (1 - \phi)
$$

#### Priors

$$
\kappa \sim \text{Beta}(20, 1.5)
$$

$$
\phi \sim \text{Beta}(2, 2)
$$

$$
\omega \sim \text{HalfNormal}(0.1)
$$

$$
\nu \sim \text{Exponential}(0.1) + 2
$$

#### Likelihood

Returns are modeled using a Student-t distribution:

$$
r_t \sim \text{StudentT}(0, \sigma_t, \nu)
$$

This allows the model to capture heavy tails and extreme returns.

#### Initialization

The initial variance is set to the stationary variance:

$$
\sigma_0^2 = \frac{\omega}{1 - \alpha - \beta}
$$

The volatility recursion is evaluated efficiently using `pytensor.scan`.

### 3. Bayesian Inference

* Posterior inference is performed using the **No-U-Turn Sampler (NUTS)**.
* Sampling configuration:

  * 2,000 draws per chain
  * 4 chains
* The result is a full posterior distribution over all model parameters.

### 4. Diagnostics

Model convergence is assessed using **ArviZ**:

* Trace plots
* $\hat{R}$ (R-hat) statistics
* Effective Sample Size (ESS)

All diagnostics indicate stable sampling and well-mixed chains.

### 5. Analysis and Visualization

* **Volatility Plot**: Posterior mean volatility $\sigma_t$ plotted alongside observed returns.
* **Posterior Predictive Check (PPC)**: Simulated return paths drawn from the posterior to assess model fit.

### 6. Forecasting and Risk Management

#### One-Step-Ahead Volatility Forecast

$$
\sigma_{t+1}^2 = \omega + \alpha r_t^2 + \beta \sigma_t^2
$$

Posterior means are used for point forecasts.

#### Risk Metrics

* **1% Value-at-Risk (VaR)**
  The loss threshold exceeded with 1% probability.

* **1% Expected Shortfall (ES)**
  The expected loss conditional on losses exceeding the 1% VaR.

Both metrics are derived from the posterior predictive distribution.

### Usage

1. Open `bayesian_garch(1,1).ipynb` in a Jupyter environment.
2. Install all dependencies.
3. Run all cells sequentially.

### Key Outputs

* Trace plots confirming convergence
* Volatility vs. returns visualization
* One-step-ahead volatility forecast
* 1% VaR and Expected Shortfall estimates


