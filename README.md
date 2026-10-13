# Advanced Econometrics III 🔍📉

*A Modern Toolkit for Cutting-Edge Econometric Analysis*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![R](https://img.shields.io/badge/R-4.3%2B-276DC3?logo=r)
![LaTeX](https://img.shields.io/badge/LaTeX-Typesetting-008080?logo=latex)

This repository contains implementations, lecture notes, and applications from an advanced graduate econometrics course, covering modern techniques beyond the classical linear model.

---

## 📚 Core Topics

### 1. **Foundations**
- Maximum Likelihood Estimation
- Numerical Optimization (Newton-Raphson, BFGS)
- Delta Method and Asymptotic Theory

### 2. **Discrete & Limited Dependent Variables**
- Logit/Probit Models
- Tobit Models
- Sample Selection Models (Heckman Correction)

### 3. **Quantile Methods**
- Least Absolute Deviations
- Quantile Regression (Koenker-Bassett)

### 4. **Causal Inference**
- Potential Outcomes Framework
- Difference-in-Differences
- Regression Discontinuity

### 5. **Nonparametric & Semiparametric**
- Kernel Regression
- Local Polynomial Estimation
- Partially Linear Models

### 6. **Specialized Models**
- Duration Analysis (Cox Proportional Hazards)
- Panel Data Methods (Fixed/Random Effects)
- Generalized Linear Models

### 7. **Computational Methods**
- Bootstrap (Parametric/Nonparametric)
- Simulation-Based Inference
- Randomization Inference

---

## 💻 Technical Implementation

```python
# Heckman Selection Model in Python
import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS

def heckman_two_step(y, X, selection, instruments):
    # First stage: Probit selection equation
    probit = Probit(selection, instruments).fit()
    mills = stats.norm.pdf(probit.fittedvalues)/stats.norm.cdf(probit.fittedvalues)
    
    # Second stage: Outcome equation with Mills ratio
    X_aug = np.column_stack([X, mills])
    return OLS(y, X_aug).fit()
```
# Heckman Selection Model in R

```r
# Heckman Two-Step Estimation in R
library(sampleSelection)

heckman_2step <- function(formula_outcome, formula_selection, data) {
  # First stage: Probit selection equation
  # Second stage: Outcome equation with inverse Mills ratio
  heckit(
    selection = formula_selection,
    outcome = formula_outcome,
    data = data,
    method = "2step"
  )
}

# Example usage:
# model <- heckman_2step(
#   formula_outcome = wage ~ education + experience,
#   formula_selection = participate ~ education + age + children,
#   data = women_data
# )
# summary(model)
```

## Execute the committed birthweight workflow

```bash
pip install -r requirements.txt
RESEARCH_MAX_ROWS=2000 python run_research.py econometrics_III.ipynb
# Full study (100,000 committed observations; more expensive):
python run_research.py econometrics_III.ipynb
```

The bounded sample is explicitly labeled smoke-only. Exact Mahalanobis matching uses whitening and a spatial index; the binary-treatment weighted quantile fit uses exact weighted group quantiles instead of a nonsmooth quasi-Newton optimization. Matching among treated observations estimates ATT, not population ATE. Identification still requires conditional exchangeability, overlap and correctly specified models; the coursework selection correction uses birthweight levels with a normal inverse Mills ratio and is not a validated structural causal estimator. Other notebooks require original missing inputs listed in REPRODUCIBILITY.md.

The selection-control exercise uses the latent probit index and the signed generalized residual for both observed treatment groups. It does not feed predicted probabilities through the normal CDF a second time. Tail ratios are evaluated in log space so extreme but finite indices remain finite.

Verification: `python -m unittest discover -s tests -v` asserts original scalar matching indices, outcomes and ATT on tied fixtures and checks random non-tied matching parity. Weighted quantiles minimize the original check-loss objective; this is a deliberate solver correction replacing potentially failed nonsmooth L-BFGS steps, not a claim that its fitted coefficients match failed iterations. At nonunique weighted quantile optima, the left observed quantile is selected deterministically.
