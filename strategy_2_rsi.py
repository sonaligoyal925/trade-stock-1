
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load market data
data = pd.read_csv("market_data.csv", index_col=0, parse_dates=True)

# Calculate RSI using 14-day average gains and losses
price_change = data["Close"].diff()

gain = price_change.clip(lower=0)
loss = -price_change.clip(upper=0)

avg_gain = gain.ewm(
    alpha=1/14, min_periods=14, adjust=False
).mean()

avg_loss = loss.ewm(
    alpha=1/14, min_periods=14, adjust=False
).mean()

relative_strength = avg_gain / avg_loss
data["RSI"] = 100 - (100 / (1 + relative_strength))

# Calculate 20-day average volume
data["Volume_Avg_20"] = data["Volume"].rolling(20).mean()

# Generate signals
data["Signal"] = 0

# Buy: RSI below 30 and volume above 1.2 times its average
buy_condition = (
    (data["RSI"] < 30) &
    (data["Volume"] > 1.2 * data["Volume_Avg_20"])
)

data.loc[buy_condition, "Signal"] = 1

# Sell: RSI crosses above 50
sell_condition = (
    (data["RSI"] > 50) &
    (data["RSI"].shift(1) <= 50)
)

data.loc[sell_condition, "Signal"] = -1

# Execute signals on the next trading day's Open
data["Execution_Signal"] = data["Signal"].shift(1).fillna(0)

# Backtest: long-only, one position at a time
capital = 100000
cash = capital
shares = 0
trades = []

for i in range(1, len(data)):
    price = data["Open"].iloc[i]
    signal = data["Execution_Signal"].iloc[i]

    if signal == 1 and shares == 0:
        shares = int(cash // price)

        if shares > 0:
            cash -= shares * price

            trades.append({
                "Date": data.index[i],
                "Action": "BUY",
                "Price": price,
                "Shares": shares
            })

    elif signal == -1 and shares > 0:
        cash += shares * price

        trades.append({
            "Date": data.index[i],
            "Action": "SELL",
            "Price": price,
            "Shares": shares
        })

        shares = 0

# Calculate final portfolio value
final_value = cash + shares * data["Close"].iloc[-1]

total_return = ((final_value - capital) / capital) * 100

# Buy-and-hold comparison
buy_hold_return = (
    (data["Close"].iloc[-1] / data["Close"].iloc[0]) - 1
) * 100

# Display results
print("RSI + High Volume Strategy")
print("--------------------------------")
print("Initial Capital: ₹", capital)
print("Final Portfolio Value: ₹", round(final_value, 2))
print("Strategy Return:", round(total_return, 2), "%")
print("Buy and Hold Return:", round(buy_hold_return, 2), "%")
print("Completed Trades:", len(trades) // 2)
print("Total Executed Orders:", len(trades))

print("\nTrade History:")
print(pd.DataFrame(trades))

# Plot price and RSI
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 9), sharex=True
)

ax1.plot(data.index, data["Close"], label="Close Price")
ax1.set_title("RELIANCE - RSI + High Volume Strategy")
ax1.set_ylabel("Price (₹)")
ax1.legend()
ax1.grid()

ax2.plot(data.index, data["RSI"], label="RSI (14)")
ax2.axhline(30, linestyle="--", label="Oversold (30)")
ax2.axhline(50, linestyle="--", label="Exit Level (50)")
ax2.axhline(70, linestyle="--", label="Overbought (70)")
ax2.set_ylabel("RSI")
ax2.set_xlabel("Date")
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.show()
