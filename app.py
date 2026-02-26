"""
Portfolio Optimizer — Streamlit Dashboard
Author: Wilfried LAWSON HELLU | github.com/Wxlly00
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from main import (get_returns, portfolio_stats, max_sharpe_portfolio,
                   min_vol_portfolio, simulate_random_portfolios, RISK_FREE_RATE)

st.set_page_config(
    page_title="Portfolio Optimizer | Wilfried LAWSON HELLU",
    page_icon="📊",
    layout="wide",
)

ALL_TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "GS", "BLK", "V", "AMZN", "NVDA", "META",
               "TSLA", "JNJ", "PG", "HD", "UNH"]

with st.sidebar:
    st.title("📊 Portfolio Optimizer")
    selected_tickers = st.multiselect(
        "Select Assets", ALL_TICKERS,
        default=["AAPL", "MSFT", "JPM", "GS", "BLK"]
    )
    period = st.select_slider("Data Period", ["1y", "2y", "3y"], value="2y")
    rf = st.slider("Risk-Free Rate (%)", 0.0, 8.0, 4.5, step=0.25) / 100
    n_sims = st.select_slider("Simulations", [500, 1000, 2000], value=1000)
    run = st.button("🚀 Optimize Portfolio", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("By [Wilfried LAWSON HELLU](https://linkedin.com/in/wilfried-lawsonhellu)")

st.title("📊 Portfolio Optimizer — Efficient Frontier")
st.markdown("*Markowitz mean-variance optimization with real market data*")

if len(selected_tickers) < 3:
    st.warning("Please select at least 3 assets")
    st.stop()

if not run:
    st.info("Configure your portfolio in the sidebar and click **Optimize Portfolio**")
    st.stop()

with st.spinner("Downloading market data and running optimization..."):
    returns = get_returns(selected_tickers, period)
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    n = len(selected_tickers)
    equal_w = np.ones(n) / n

    ms_w, ms_r, ms_v, ms_s = max_sharpe_portfolio(mean_returns, cov_matrix, rf)
    mv_w, mv_r, mv_v, mv_s = min_vol_portfolio(mean_returns, cov_matrix)
    eq_r, eq_v, eq_s = portfolio_stats(equal_w, mean_returns, cov_matrix, rf)
    sims = simulate_random_portfolios(mean_returns, cov_matrix, n=n_sims, rf=rf)

st.success(f"✓ Optimized {len(selected_tickers)} assets | {len(returns)} trading days")

tab1, tab2, tab3 = st.tabs(["🎯 Optimization", "📈 Efficient Frontier", "📊 Performance"])

with tab1:
    st.subheader("Optimal Portfolio Weights")
    c1, c2, c3 = st.columns(3)
    c1.metric("Max Sharpe Return", f"{ms_r*100:.1f}%")
    c1.metric("Max Sharpe Sharpe", f"{ms_s:.2f}")
    c2.metric("Max Sharpe Vol", f"{ms_v*100:.1f}%")
    c3.metric("Min Vol Return", f"{mv_r*100:.1f}%")
    c3.metric("Min Vol Volatility", f"{mv_v*100:.1f}%")

    # Weights comparison
    weights_df = pd.DataFrame({
        "Asset": selected_tickers,
        "Max Sharpe (%)": (ms_w * 100).round(1),
        "Min Vol (%)": (mv_w * 100).round(1),
        "Equal Weight (%)": [round(100/n, 1)] * n,
    }).sort_values("Max Sharpe (%)", ascending=False)
    st.dataframe(weights_df, use_container_width=True, hide_index=True)

    # Pie chart
    col_a, col_b = st.columns(2)
    for col, title, weights in [(col_a, "Max Sharpe", ms_w), (col_b, "Min Volatility", mv_w)]:
        fig_pie = go.Figure(go.Pie(
            labels=selected_tickers, values=weights * 100,
            hole=0.4, textinfo="label+percent",
            marker=dict(colors=px.colors.qualitative.Set2),
        ))
        fig_pie.update_layout(
            title=title, paper_bgcolor="#050D1A",
            font_color="#94A3B8", title_font_color="white",
            showlegend=False, height=350,
        )
        col.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.subheader("Efficient Frontier")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sims["Volatility"], y=sims["Return"],
        mode="markers",
        marker=dict(
            size=4, color=sims["Sharpe"],
            colorscale="YlOrRd", opacity=0.5,
            colorbar=dict(title="Sharpe", tickfont=dict(color="#94A3B8")),
        ),
        name="Random Portfolios",
        text=[f"Sharpe: {s:.2f}" for s in sims["Sharpe"]],
        hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<br>%{text}",
    ))
    fig.add_trace(go.Scatter(
        x=[ms_v * 100], y=[ms_r * 100], mode="markers",
        marker=dict(size=18, symbol="star", color="#C9A84C", line=dict(width=1, color="white")),
        name=f"Max Sharpe ({ms_s:.2f})",
    ))
    fig.add_trace(go.Scatter(
        x=[mv_v * 100], y=[mv_r * 100], mode="markers",
        marker=dict(size=14, symbol="square", color="#4CAF50", line=dict(width=1, color="white")),
        name=f"Min Volatility ({mv_v*100:.1f}%)",
    ))

    fig.update_layout(
        paper_bgcolor="#050D1A", plot_bgcolor="#0A1628",
        font_color="#94A3B8",
        xaxis=dict(title="Annual Volatility (%)", gridcolor="#1E2D45"),
        yaxis=dict(title="Annual Expected Return (%)", gridcolor="#1E2D45"),
        title="Efficient Frontier — Mean-Variance Space",
        title_font_color="white", height=500,
        legend=dict(bgcolor="#0A1628", bordercolor="#C9A84C", font=dict(color="white")),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Cumulative Returns Comparison")
    
    equal_weights = np.ones(n) / n
    ms_returns_series = (returns * ms_w).sum(axis=1)
    mv_returns_series = (returns * mv_w).sum(axis=1)
    eq_returns_series = (returns * equal_weights).sum(axis=1)
    
    ms_cumulative = (1 + ms_returns_series).cumprod() * 100
    mv_cumulative = (1 + mv_returns_series).cumprod() * 100
    eq_cumulative = (1 + eq_returns_series).cumprod() * 100
    
    fig3 = go.Figure()
    for cumret, label, color in [
        (ms_cumulative, "Max Sharpe", "#C9A84C"),
        (mv_cumulative, "Min Volatility", "#4CAF50"),
        (eq_cumulative, "Equal Weight", "#4A6FA5"),
    ]:
        fig3.add_trace(go.Scatter(
            x=cumret.index, y=cumret.values,
            name=label, line=dict(color=color, width=2),
        ))
    
    fig3.update_layout(
        paper_bgcolor="#050D1A", plot_bgcolor="#0A1628",
        font_color="#94A3B8",
        xaxis=dict(gridcolor="#1E2D45"),
        yaxis=dict(title="Portfolio Value (base = 100)", gridcolor="#1E2D45"),
        title="Cumulative Return: Optimized vs Equal Weight",
        title_font_color="white", height=420,
        legend=dict(bgcolor="#0A1628", bordercolor="#C9A84C", font=dict(color="white")),
    )
    st.plotly_chart(fig3, use_container_width=True)

st.caption("Built by **Wilfried LAWSON HELLU** | Finance Analyst | github.com/Wxlly00")
