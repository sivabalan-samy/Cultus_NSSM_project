# Submission: Advanced Time Series Forecasting with Neural State Space Models (NSSMs)

## **Project Overview**
This project implements a **Neural State Space Model (NSSM)** and compares it with a baseline **LSTM** model for multivariate time-series forecasting.  
A synthetic dataset is generated using numpy with:

- Clear trend  
- Seasonality  
- Autoregressive (AR) behavior  
- Gaussian noise  
- 5 features  
- 1000 data points  

The aim is to learn latent state transitions and evaluate forecasting performance against a standard LSTM.

---

## **Dataset Generation**
The dataset is generated programmatically using:

```python
generate_multivariate_series(seq_len=1000, n_features=5)
