# Modern Portfolio Theory (MPT) with Indian Stocks

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Step 1: Fetch stock data (Indian stocks)
stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

# Configurable start and end dates
start_date = "2020-01-01"
end_date = "2025-08-31"  # Change this variable as needed

# Download stock and NIFTY50 data
data = yf.download(stocks, start=start_date, end=end_date)["Adj Close"]
nifty = yf.download("^NSEI", start=start_date, end=end_date)["Adj Close"]

# Step 2: Compute returns
daily_returns = data.pct_change().dropna()
mean_returns = daily_returns.mean()
cov_matrix = daily_returns.cov()

nifty_returns = nifty.pct_change().dropna()

# Step 3: Portfolio simulation with constraints
num_portfolios = 50000
results = np.zeros((3, num_portfolios))
weights_record = []
risk_free_rate = 0.06 / 252  # Assuming ~6% annual RF rate in India, daily

for i in range(num_portfolios):
    # Generate random weights with constraint: no stock > 30%
    weights = np.random.dirichlet(np.ones(len(stocks)), size=1).flatten()
    while np.any(weights > 0.3):
        weights = np.random.dirichlet(np.ones(len(stocks)), size=1).flatten()

    weights_record.append(weights)

    portfolio_return = np.sum(weights * mean_returns) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (portfolio_return - 0.06) / portfolio_volatility

    results[0, i] = portfolio_return
    results[1, i] = portfolio_volatility
    results[2, i] = sharpe_ratio

# Step 4: Convert to DataFrame
results_df = pd.DataFrame(results.T, columns=["Return", "Volatility", "Sharpe"])
weights_df = pd.DataFrame(weights_record, columns=stocks)
portfolio_df = pd.concat([results_df, weights_df], axis=1)

# Step 5: Identify optimal portfolios
max_sharpe_portfolio = portfolio_df.loc[portfolio_df["Sharpe"].idxmax()]
min_vol_portfolio = portfolio_df.loc[portfolio_df["Volatility"].idxmin()]

# Step 6: Visualization
plt.figure(figsize=(12, 8))
plt.scatter(
    portfolio_df["Volatility"],
    portfolio_df["Return"],
    c=portfolio_df["Sharpe"],
    cmap="viridis",
    alpha=0.5,
)
plt.colorbar(label="Sharpe Ratio")

# Highlight optimal portfolios
plt.scatter(
    max_sharpe_portfolio["Volatility"],
    max_sharpe_portfolio["Return"],
    c="red",
    s=200,
    marker="*",
    label="Max Sharpe",
)
plt.scatter(
    min_vol_portfolio["Volatility"],
    min_vol_portfolio["Return"],
    c="blue",
    s=200,
    marker="*",
    label="Min Volatility",
)

plt.xlabel("Volatility (Risk)")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier - Indian Stocks (With Constraints)")
plt.legend()
plt.show()

# Step 7: Backtesting against NIFTY50
# Build time-series of optimized portfolio (Max Sharpe)
portfolio_daily_returns = (daily_returns @ max_sharpe_portfolio[stocks])

cumulative_portfolio = (1 + portfolio_daily_returns).cumprod()
cumulative_nifty = (1 + nifty_returns).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(cumulative_portfolio, label="Optimized Portfolio (Max Sharpe)")
plt.plot(cumulative_nifty, label="NIFTY50")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.title(f"Backtest: Portfolio vs NIFTY50 ({start_date} - {end_date})")
plt.legend()
plt.show()

# Display optimal portfolios
print("Maximum Sharpe Ratio Portfolio:\n", max_sharpe_portfolio)
print("\nMinimum Volatility Portfolio:\n", min_vol_portfolio)
