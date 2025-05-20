import streamlit as st
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Optional: Riskfolio for portfolio optimization
try:
    import riskfolio.Portfolio as pf
    riskfolio_available = True
except ImportError:
    riskfolio_available = False

st.set_page_config(page_title="Stock Correlation & Risk", layout="wide")
st.title("📈 Stock Correlation & Risk Dashboard")

# Sidebar controls
st.sidebar.header("Configuration")
ticker_options = ["ANET", "FN", "ALAB", "AAPL", "TSLA", "MSFT", "AMZN", "NVDA"]
tickers = st.sidebar.multiselect("Select tickers", options=ticker_options, default=ticker_options)
start = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2025-04-30"))
window = st.sidebar.slider("Rolling Correlation Window (days)", 20, 180, 60)

# Run analysis
if st.sidebar.button("🔍 Run Analysis"):
    with st.spinner("Fetching and analyzing data..."):
        data = {}

        for ticker in tickers:
            st.write(f"📥 Downloading {ticker}...")
            stock = yf.download(ticker, start=start, end=end)

            if stock.empty:
                st.error(f"❌ No data found for {ticker}.")
                continue

            # Handle MultiIndex if present
            if isinstance(stock.columns, pd.MultiIndex):
                try:
                    close_prices = stock["Close"][ticker].dropna()
                except KeyError:
                    st.warning(f"⚠️ 'Close' prices not found for {ticker}.")
                    continue
            else:
                if "Close" not in stock.columns:
                    st.warning(f"⚠️ 'Close' prices not found for {ticker}.")
                    continue
                close_prices = stock["Close"].dropna()

            if not close_prices.empty:
                data[ticker] = close_prices
            else:
                st.warning(f"⚠️ No valid Close prices for {ticker}.")

        if not data:
            st.error("❌ No valid data downloaded.")
        else:
            df = pd.DataFrame(data).ffill().dropna()

            st.subheader("📊 Combined Close Prices")
            st.dataframe(df.tail(10))
            st.line_chart(df)

            # CSV download
            csv = df.to_csv(index=True).encode("utf-8")
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="close_prices.csv",
                mime="text/csv"
            )

            # Correlation matrix
            returns = df.pct_change().dropna()
            corr = returns.corr()

            st.subheader("📌 Correlation Matrix")
            st.dataframe(corr.round(3))

            st.subheader("🔴 Heatmap")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
            st.pyplot(fig)

            # Download correlation matrix
            buffer = io.StringIO()
            corr.to_csv(buffer)
            st.download_button(
                "⬇️ Download Correlation Matrix CSV",
                data=buffer.getvalue().encode("utf-8"),
                file_name="correlation_matrix.csv",
                mime="text/csv"
            )

            # Rolling correlation
            if len(tickers) >= 2:
                st.subheader(f"🔁 Rolling Correlation: {tickers[0]} vs {tickers[1]}")
                roll_corr = returns[tickers[0]].rolling(window).corr(returns[tickers[1]])
                st.line_chart(roll_corr.dropna())

            # Portfolio optimization
            if riskfolio_available and len(tickers) > 1:
                st.subheader("📉 Risk Metrics (VaR, CVaR, Sharpe)")
                port = pf.Portfolio(returns=returns)
                port.assets_stats(method_mu='hist', method_cov='hist')
                risk = port.risk_measures(method='hist', rf=0)
                st.dataframe(risk[["VaR_0.05", "CVaR_0.05", "Sharpe"]].round(4))

                st.subheader("🧠 Portfolio Optimization (Max Sharpe)")
                w = port.optimization(model="Classic", rm="MV", obj="Sharpe", hist=True)
                st.dataframe(w.T.round(4))

                port_weights = w[w > 0].index.tolist()
                weighted_returns = returns[port_weights].mul(w.T[port_weights].values, axis=1).sum(axis=1)
                cumulative_returns = (1 + weighted_returns).cumprod()
                st.subheader("📈 Optimized Portfolio Cumulative Returns")
                st.line_chart(cumulative_returns)

                st.subheader("📉 Drawdown Chart")
                drawdown = (cumulative_returns - cumulative_returns.cummax()) / cumulative_returns.cummax()
                st.line_chart(drawdown)

                st.subheader("🔁 Monthly Rebalanced Portfolio Returns")
                rebalance_returns = weighted_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                st.line_chart((1 + rebalance_returns).cumprod())
            elif not riskfolio_available:
                st.warning("⚠️ Install `riskfolio-lib` to enable risk metrics and optimization.")
    st.success("✅ Analysis complete!")
else:
    st.info("👈 Select tickers and press 'Run Analysis' to begin.")