# Portfolio Optimizer

A transparent mean-variance portfolio construction tool for exploring allocation trade-offs, efficient portfolios and risk-adjusted return using Python.

> **Portfolio case study:** this repository is designed as an analytical portfolio project. It is not investment advice, a recommendation to buy or sell securities, or a production asset-allocation engine.

## Why This Project

Multi-asset portfolio management requires a structured way to compare expected return, volatility and diversification across a set of assets. This project implements a classic Markowitz framework to show how portfolio weights change when the objective shifts from minimum volatility to maximum Sharpe ratio.

The emphasis is on three things:

1. **Portfolio construction** — translating return and covariance estimates into constrained asset weights.
2. **Risk / return trade-offs** — visualising the efficient set rather than looking at assets individually.
3. **Transparent assumptions** — making the optimisation objective, constraints and data source explicit.

## Core Outputs

The model produces:

- **Maximum Sharpe Ratio portfolio**
- **Minimum Volatility portfolio**
- **Efficient-frontier visualisation**
- **1,000 random long-only portfolios** for comparison
- **Annualised expected return, volatility and Sharpe ratio**
- **Portfolio weights by asset**

The Streamlit interface allows the user to select the investment universe, change the risk-free rate and explore the resulting allocation interactively.

## Methodology

For portfolio weights \(w\), expected returns \(\mu\) and covariance matrix \(\Sigma\):

```text
Expected portfolio return = w'μ
Portfolio variance        = w'Σw
Portfolio volatility      = √(w'Σw)
Sharpe ratio              = (Rp − Rf) / σp
```

The implementation annualises daily return and covariance estimates using 252 trading days.

### Optimisation constraints

The current model uses:

- weights summing to 100%;
- long-only positions;
- individual weights bounded between 0% and 100%;
- SLSQP optimisation through `scipy.optimize.minimize`.

Two optimisation problems are solved:

**Maximum Sharpe Ratio**

```text
maximise (Expected Return − Risk-Free Rate) / Volatility
```

**Minimum Volatility**

```text
minimise Portfolio Volatility
```

## Data

The default implementation requests approximately two years of adjusted market data through `yfinance` and computes daily percentage returns.

If live data cannot be retrieved, the code can generate **explicitly synthetic correlated returns** so the application remains demonstrable. Synthetic data is only a fallback for illustration and must not be interpreted as observed market history.

Default universe:

`AAPL · MSFT · GOOGL · JPM · GS · BLK · V · AMZN · NVDA · META`

## Asset-Management Relevance

This project demonstrates several building blocks used in portfolio-management workflows:

- allocation across multiple securities;
- covariance-based diversification analysis;
- risk-adjusted return comparison;
- optimisation under portfolio constraints;
- visual communication of portfolio trade-offs;
- integration of financial data, numerical methods and an analyst-facing interface.

The project is intentionally compact. It focuses on making the mechanics readable rather than presenting the framework as a complete institutional allocation process.

## Tech Stack

| Area | Library |
|---|---|
| Market data | `yfinance` |
| Data analysis | `pandas`, `numpy` |
| Optimisation | `scipy.optimize` |
| Visualisation | `matplotlib`, `plotly` |
| Interface | `streamlit` |

## Repository Structure

```text
portfolio-optimizer/
├── main.py             # Core calculations and efficient-frontier workflow
├── app.py              # Streamlit interface
├── requirements.txt
└── README.md
```

## How to Run

```bash
git clone https://github.com/Wxlly00/portfolio-optimizer.git
cd portfolio-optimizer
pip install -r requirements.txt
```

Run the analytical workflow:

```bash
python main.py
```

Run the interactive dashboard:

```bash
streamlit run app.py
```

## Assumptions and Limitations

The project uses a deliberately simple mean-variance framework. Important limitations include:

- expected returns are estimated from a relatively short historical window;
- the sample covariance matrix is used without shrinkage or Bayesian adjustment;
- optimisation is sensitive to estimation error, especially expected returns;
- no transaction costs, turnover penalties, taxes or liquidity constraints are modelled;
- no benchmark-relative objective, tracking error or active-risk constraint is included;
- no short selling, leverage, minimum position size or asset-class exposure limits are included;
- the framework assumes historical covariance is informative for future risk;
- returns are treated through a static single-period optimisation rather than a dynamic allocation process;
- synthetic fallback data is illustrative only.

A production multi-asset process would typically add stronger covariance estimation, strategic/tactical allocation constraints, benchmark-relative risk, turnover controls, scenario analysis, stress testing and out-of-sample validation.

## Author

**Wilfried LAWSON HELLU**  
**Finance × Data × Software**

[GitHub](https://github.com/Wxlly00)
