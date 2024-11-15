# Import required libraries
import pandas as pd
import numpy as np
import yfinance as yf  # To fetch stock data
from statsmodels.regression.rolling import RollingOLS # type: ignore
from scipy.optimize import minimize # type: ignore

# Defined parameters
tickers_long = [
    "CEG", "LH", "BA", "CARR", "DOW", "PH", "EMR", "JBL", "SWK", "URI", "BSX", "DLTR", "ORCL", "HUBB", "LYB",
    "XYL", "HON", "DD", "ROP", "UNH", "IBM", "GRMN", "CMI", "BKR", "GLW", "SYK", "FTV", "ETN", "CHD", "OTIS",
    "PCAR", "DGX", "AME", "DRI", "APH", "AOS", "HUM", "CLX", "ORLY", "CTAS", "ECL", "TER", "TMUS", "MAS", "TDG",
    "JNPR", "NSC", "FAST", "PAYX", "ROK", "ITW", "CSCO", "CPRT", "TMO", "OKE", "EXC", "EMN", "PWR", "NEM", "DOV",
    "VTR", "TXT", "TXN", "PG", "AVY", "DTE", "MGM", "BR", "GD", "ADP", "PPL", "NI", "MLM", "IDXX", "HCA", "SHW",
    "HWM", "ZTS", "RTX", "UNP", "MCK", "AES", "FICO", "INTC", "JCI", "ATO", "HAS", "LOW", "ALLE", "WELL", "ISRG",
    "VRSN", "TRGP", "LMT"
]
tickers_short = [
    "ETSY", "DXCM", "ILMN", "PAYC", "VFC", "ABNB", "APA", "UPS", "EPAM", "CHTR", "MOS", "EXPE", "MPC", "PANW",
    "VLO", "COR", "BXP", "MRO", "HAL", "MRNA"
]

start_date = "2022-01-01"
end_date = "2024-11-10"
trailing_stop = 0.03  # 3% trailing stop loss to minimize risk

# Step 1: Fetch historical price data
data = yf.download(tickers_long + tickers_short, start=start_date, end=end_date)['Adj Close']
returns = data.pct_change().dropna()

# Step 2: Calculate rolling beta for each stock
market_returns = returns.mean(axis=1)  # Equal-weighted market index approximation
betas = {}

for ticker in tickers_long + tickers_short:
    model = RollingOLS(returns[ticker], market_returns, window=252)
    betas[ticker] = model.fit().params

# Average beta for each stock
average_betas = {ticker: np.mean(betas[ticker]) for ticker in tickers_long + tickers_short}

# Step 3: Define Portfolio Optimization with Target Return Constraint
# Target an aggresive annualized return of 15%
target_annual_return = 0.15
target_daily_return = target_annual_return / 252  # Approximate daily return target

# Objective function to maximize Sharpe ratio
def negative_sharpe_ratio(weights, returns):
    portfolio_return = np.dot(weights, returns.mean())
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
    return -portfolio_return / portfolio_volatility  # Negative for minimization

# Constraints and bounds
# - Sum of weights must be 1
# - Portfolio return must meet or exceed target_daily_return
constraints = [
    {"type": "eq", "fun": lambda weights: np.sum(weights) - 1},
    {"type": "ineq", "fun": lambda weights: np.dot(weights, returns.mean()) - target_daily_return}
]
bounds = [(0, 1) for _ in range(len(tickers_long + tickers_short))]  # Allow higher allocations

# Initial weights for optimization
initial_weights = np.array([1 / len(tickers_long + tickers_short)] * len(tickers_long + tickers_short))

# Step 4: Run the optimization
opt_result = minimize(negative_sharpe_ratio, initial_weights, args=(returns,), bounds=bounds, constraints=constraints)
optimized_weights = opt_result.x

# Step 5: Apply trailing stop loss function
def apply_trailing_stop_loss(returns, trailing_stop):
    stop_loss_returns = returns.copy()
    for ticker in returns.columns:
        max_price = returns[ticker].cummax()
        drawdown = (returns[ticker] - max_price) / max_price
        stop_loss_returns[ticker][drawdown < -trailing_stop] = 0  # Set returns to 0 for breached stop-loss
    return stop_loss_returns

# Adjusted returns after applying trailing stop loss
adjusted_returns = apply_trailing_stop_loss(returns, trailing_stop)

# Step 6: Calculate portfolio performance metrics
final_portfolio_return = np.dot(optimized_weights, adjusted_returns.mean()) * 252  # Annualize the return
final_portfolio_volatility = np.sqrt(np.dot(optimized_weights.T, np.dot(adjusted_returns.cov(), optimized_weights)) * 252)  # Annualize the volatility

# Output the results
print("Optimized Weights:", optimized_weights)
print("Final Portfolio Return (Annualized):", final_portfolio_return)
print("Final Portfolio Volatility (Annualized):", final_portfolio_volatility)

