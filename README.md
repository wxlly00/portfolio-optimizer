# Portfolio Optimizer

> Markowitz mean-variance portfolio optimization with interactive Streamlit dashboard

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)](https://streamlit.io)

## Overview

Professional portfolio optimization framework implementing Modern Portfolio Theory. Finds the optimal asset allocation that maximizes the Sharpe ratio or minimizes volatility for a given universe of stocks.

## Methodology

### Efficient Frontier (Markowitz, 1952)
- Expected returns estimated from historical daily returns × 252
- Covariance matrix computed from historical price data
- Monte Carlo simulation of 1,000+ random portfolios to map the feasible space
- Optimal portfolios found via `scipy.optimize.minimize` (SLSQP)

### Optimization Objectives
| Portfolio | Objective |
|-----------|-----------|
| **Max Sharpe** | Maximize `(E[R] - Rf) / σ` |
| **Min Volatility** | Minimize `σ = √(wᵀΣw)` |
| **Equal Weight** | Benchmark: 1/N allocation |

## Default Universe (10 US stocks)
AAPL, MSFT, GOOGL, JPM, GS, BLK, V, AMZN, NVDA, META

## Tech Stack

`Python 3.11` `Streamlit` `PyPortfolioOpt` `yfinance` `scipy` `Plotly` `matplotlib`

## How to Run

```bash
pip install -r requirements.txt

# Terminal + chart
python main.py

# Interactive dashboard
streamlit run app.py
```

## Output

- Optimal portfolio weights for 3 strategies
- Annual return, volatility, Sharpe ratio comparison
- Efficient frontier PNG (`output/efficient_frontier.png`)
- Interactive Streamlit dashboard with portfolio analytics

## Author

**Wilfried LAWSON HELLU** | Finance Analyst  
📧 wilfriedlawpro@gmail.com | 🔗 [LinkedIn](https://linkedin.com/in/wilfried-lawsonhellu) | 🐙 [GitHub](https://github.com/Wxlly00)
