# Portfolio Optimizer

> Markowitz Mean-Variance Optimization with efficient frontier visualization.

## Description

A quantitative portfolio construction tool that applies **Modern Portfolio Theory** to find optimal asset allocations. Computes the efficient frontier, identifies the Max Sharpe Ratio portfolio and the Minimum Volatility portfolio, and renders an interactive chart via Plotly/Streamlit.

## Tech Stack

| Layer | Library |
|-------|---------|
| Data  | `yfinance` |
| Numerics | `numpy`, `pandas` |
| Optimisation | `scipy.optimize` |
| Visualisation | `plotly`, `matplotlib`, `streamlit` |

## Installation

```bash
git clone https://github.com/Wxlly00/portfolio-optimizer.git
cd portfolio-optimizer
pip install -r requirements.txt
```

## Usage

### CLI
```bash
python main.py            # Runs optimization on default 10-asset universe
```

### Streamlit App
```bash
streamlit run app.py
```

Select tickers, adjust the risk-free rate, and explore the efficient frontier interactively.

## Features

- **Efficient frontier** — 1,000 random portfolios + optimized boundary
- **Max Sharpe Ratio** portfolio — highest risk-adjusted return
- **Min Volatility** portfolio — lowest annualized standard deviation
- **Live data** via yfinance (2-year history) with synthetic fallback
- Interactive scatter plot with color-coded Sharpe ratios

## Default Universe

`AAPL · MSFT · GOOGL · JPM · GS · BLK · V · AMZN · NVDA · META`

## Author

**Wilfried LAWSON HELLU** — Finance Analyst  
[github.com/Wxlly00](https://github.com/Wxlly00)
