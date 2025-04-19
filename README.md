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
