# mpt-indian-stocks
Modern Portfolio Theory with Indian Stocks in Python


This is a Python project where I applied **Modern Portfolio Theory (MPT)** to Indian stocks.  
The goal is to figure out the best portfolio by balancing **risk** and **return**, and then compare it with the **NIFTY50 index**.

- Pulls historical data of Indian stocks using Yahoo Finance.  
- Calculates **expected returns, volatility, and Sharpe Ratio**.  
- Runs thousands of simulations to build an **Efficient Frontier**.  
- Finds the two most interesting portfolios:
  - ⭐ The one with the **highest Sharpe Ratio** (best risk-adjusted returns).  
  - 🔵 The one with the **lowest volatility** (least risky).  
- Backtests the optimized portfolio against **NIFTY50** to see how it performs.  
- Adds a realistic constraint: no single stock can be more than **30% of the portfolio**.  

 Built With
- Python 3  
- NumPy  
- Pandas  
- Matplotlib  
- yFinance  
