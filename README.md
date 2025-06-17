# Robust Stock Index Return Prediction Using Conditional Machine Learning

This project implements and evaluates the methodology proposed in the paper  
**"Robust Stock Index Return Predictions Using Deep Learning"** by Jagannathan et al. (2023).  
We replicate their Conditional Machine Learning (CML) approach to forecast stock index returns, using deep neural networks and local PCA to extract robust cross-sectional signals from firm-level characteristics.

## Project Summary

Unlike traditional time-series forecasting models, this approach leverages:
- **Firm-level expected returns estimated via cross-sectional neural networks**
- **Local principal component analysis (local PCA)** to uncover latent stock return factors
- **Instrumental variable techniques** to estimate valuation-based (BM) factors
- **Short-term dynamic relationships** among firm characteristics and market returns

We compare this methodology with naive cross-sectional DNN and pooled ML approaches, showing the superior performance of CML in capturing conditional expected returns.

## Methodology Overview

1. **Cross-sectional DNN estimation** of expected returns from firm characteristics.
2. **Local PCA** applied to expected returns to estimate time-varying stock betas.  
   > *Note: Our implementation differs from the original paper in the kernel weighting strategy.  
   We use an **exponential decay weighting scheme** instead of the symmetric quartic kernel used in the paper.*
3. **BM-factor construction** using an IV estimator based on stock betas.
4. **Market return forecast** using estimated BM-factors.

## Datasets

- **87-characteristic dataset** from Chen & Zimmermann (2021)  
- **45-characteristic cleaned dataset** from Bryzgalova & Pelger (2025)

These include information on valuation, profitability, risk, liquidity, and more. All characteristics are rank-transformed monthly, drawn from CRSP and Compustat.

## Implementation

- Language: **Python**
- Libraries: `PyTorch`, `scikit-learn`, `statsmodels`
- Features:
  - Two-layer feedforward neural network
  - Local PCA with exponential kernel weighting
  - Out-of-sample forecasting and performance evaluation (`R²`, MSE)

## Results

Our results show that the CML approach:
- Improves out-of-sample forecast accuracy
- Reduces idiosyncratic noise
- Captures meaningful short-term dynamics in market returns

## Future Work

- Explore hybrid deep learning + time series models (e.g. LSTM, Transformers)
- Evaluate tree-based methods (e.g. LightGBM) for cross-sectional modeling
- Apply AutoML to tune model parameters and architecture

## Contributors

- Bocheng Dai  
- Yifan Geng  
- James Liu  
- Tiankai Yan

> Stanford University – MS&E 349: Financial Statistics(Spring 2025)

## Reference

- Jagannathan, R., Liao, Y., & Neuhierl, A. (2023). *Robust stock index return predictions using deep learning*. SSRN: [4890466](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4890466)
