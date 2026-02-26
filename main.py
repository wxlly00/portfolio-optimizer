"""
Portfolio Optimizer — Markowitz Mean-Variance Optimization
Author: Wilfried LAWSON HELLU | Finance Analyst
GitHub: github.com/Wxlly00

Features:
- Historical return & covariance estimation
- Max Sharpe ratio portfolio
- Min volatility portfolio
- Efficient frontier visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize
import os
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "GS", "BLK", "V", "AMZN", "NVDA", "META"]
RISK_FREE_RATE = 0.045  # 4.5%
N_PORTFOLIOS = 1000     # Random portfolio simulations for frontier


def get_returns(tickers: list, period: str = "2y") -> pd.DataFrame:
    """Download price data and compute log returns."""
    if YFINANCE_AVAILABLE:
        try:
            data = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
            if isinstance(data, pd.Series):
                data = data.to_frame(tickers[0])
            returns = data.pct_change().dropna()
            if len(returns) > 100:
                return returns
        except Exception:
            pass

    # Synthetic fallback
    np.random.seed(42)
    n_days = 504
    n_assets = len(tickers)
    
    # Approximate realistic correlations
    corr_strength = 0.6
    base_corr = corr_strength * np.ones((n_assets, n_assets)) + (1 - corr_strength) * np.eye(n_assets)
    L = np.linalg.cholesky(base_corr)
    
    vols = np.random.uniform(0.015, 0.030, n_assets)
    means = np.random.uniform(0.0002, 0.0008, n_assets)
    
    z = np.random.standard_normal((n_days, n_assets))
    raw = z @ L.T
    ret_data = raw * vols + means
    
    return pd.DataFrame(ret_data, columns=tickers)


def portfolio_stats(weights: np.ndarray, mean_returns: np.ndarray,
                    cov_matrix: np.ndarray, rf: float = RISK_FREE_RATE) -> tuple:
    """Return (annual_return, annual_vol, sharpe)."""
    weights = np.array(weights)
    ret = np.dot(weights, mean_returns) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = (ret - rf) / vol
    return ret, vol, sharpe


def max_sharpe_portfolio(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                          rf: float = RISK_FREE_RATE) -> tuple:
    """Maximize Sharpe ratio via scipy minimize."""
    n = len(mean_returns)
    
    def neg_sharpe(w):
        r, v, s = portfolio_stats(w, mean_returns, cov_matrix, rf)
        return -s

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n
    
    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result.x
    ret, vol, sharpe = portfolio_stats(weights, mean_returns, cov_matrix, rf)
    return weights, ret, vol, sharpe


def min_vol_portfolio(mean_returns: np.ndarray, cov_matrix: np.ndarray) -> tuple:
    """Minimize portfolio volatility."""
    n = len(mean_returns)
    
    def portfolio_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix * 252, w)))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n
    
    result = minimize(portfolio_vol, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result.x
    ret, vol, sharpe = portfolio_stats(weights, mean_returns, cov_matrix)
    return weights, ret, vol, sharpe


def simulate_random_portfolios(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                                n: int = N_PORTFOLIOS, rf: float = RISK_FREE_RATE) -> pd.DataFrame:
    """Monte Carlo simulation of random portfolios."""
    n_assets = len(mean_returns)
    results = {"Return": [], "Volatility": [], "Sharpe": [], "Weights": []}
    
    for _ in range(n):
        w = np.random.random(n_assets)
        w /= w.sum()
        r, v, s = portfolio_stats(w, mean_returns, cov_matrix, rf)
        results["Return"].append(r * 100)
        results["Volatility"].append(v * 100)
        results["Sharpe"].append(s)
        results["Weights"].append(w)
    
    return pd.DataFrame(results)


def plot_efficient_frontier(simulations: pd.DataFrame, max_sharpe: tuple,
                             min_vol: tuple, tickers: list, save_path: str = None):
    """Plot the efficient frontier with optimal portfolios."""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#050D1A")
    ax.set_facecolor("#0A1628")
    
    # Random portfolios colored by Sharpe
    scatter = ax.scatter(
        simulations["Volatility"], simulations["Return"],
        c=simulations["Sharpe"], cmap="YlOrRd", alpha=0.5, s=8, zorder=1
    )
    plt.colorbar(scatter, ax=ax, label="Sharpe Ratio").ax.yaxis.label.set_color("white")
    
    ms_w, ms_r, ms_v, ms_s = max_sharpe
    mv_w, mv_r, mv_v, mv_s = min_vol
    
    # Max Sharpe
    ax.scatter(ms_v * 100, ms_r * 100, marker="★", color="#C9A84C", s=300, zorder=5,
               label=f"Max Sharpe: {ms_s:.2f}")
    # Min Vol
    ax.scatter(mv_v * 100, mv_r * 100, marker="■", color="#4CAF50", s=150, zorder=5,
               label=f"Min Vol: σ={mv_v*100:.1f}%")
    
    ax.set_xlabel("Annual Volatility (%)", color="white", fontsize=11)
    ax.set_ylabel("Annual Expected Return (%)", color="white", fontsize=11)
    ax.set_title("Efficient Frontier — Markowitz Portfolio Optimization",
                  color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#1E2D45")
    ax.spines["left"].set_color("#1E2D45")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#0A1628", edgecolor="#C9A84C", labelcolor="white", fontsize=9)
    
    # Caption
    ax.text(0.01, 0.01, "By Wilfried LAWSON HELLU | github.com/Wxlly00",
            transform=ax.transAxes, color="#64748B", fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  ✓ Chart saved: {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    print("=" * 65)
    print("  PORTFOLIO OPTIMIZER — Markowitz Mean-Variance")
    print("  Author: Wilfried LAWSON HELLU | github.com/Wxlly00")
    print("=" * 65)
    
    print(f"\n  Assets: {', '.join(TICKERS)}")
    returns = get_returns(TICKERS)
    print(f"  Data: {len(returns)} trading days")
    
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    equal_w = np.ones(len(TICKERS)) / len(TICKERS)
    
    # Optimize
    ms_w, ms_r, ms_v, ms_s = max_sharpe_portfolio(mean_returns, cov_matrix)
    mv_w, mv_r, mv_v, mv_s = min_vol_portfolio(mean_returns, cov_matrix)
    eq_r, eq_v, eq_s = portfolio_stats(equal_w, mean_returns, cov_matrix)
    
    print("\n  PORTFOLIO COMPARISON")
    print(f"  {'Portfolio':<20} {'Ann. Return':>12} {'Ann. Vol':>10} {'Sharpe':>8}")
    print("  " + "-" * 55)
    print(f"  {'Max Sharpe':<20} {ms_r*100:>11.1f}% {ms_v*100:>9.1f}% {ms_s:>8.2f}")
    print(f"  {'Min Volatility':<20} {mv_r*100:>11.1f}% {mv_v*100:>9.1f}% {mv_s:>8.2f}")
    print(f"  {'Equal Weight':<20} {eq_r*100:>11.1f}% {eq_v*100:>9.1f}% {eq_s:>8.2f}")
    
    print("\n  MAX SHARPE PORTFOLIO WEIGHTS:")
    weights_df = pd.DataFrame({"Ticker": TICKERS, "Weight (%)": (ms_w * 100).round(1)})
    weights_df = weights_df[weights_df["Weight (%)"] > 0.1].sort_values("Weight (%)", ascending=False)
    print(weights_df.to_string(index=False))
    
    # Simulate & plot
    print("\n  Simulating 1,000 random portfolios...")
    sims = simulate_random_portfolios(mean_returns, cov_matrix)
    
    print("  Generating efficient frontier chart...")
    plot_efficient_frontier(
        sims,
        (ms_w, ms_r, ms_v, ms_s),
        (mv_w, mv_r, mv_v, mv_s),
        TICKERS,
        save_path="output/efficient_frontier.png",
    )
    
    print("\n  ✅ Portfolio optimization complete!")


if __name__ == "__main__":
    main()
