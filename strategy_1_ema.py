
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load market data
data = pd.read_csv("market_data.csv", index_col=0, parse_dates=True)

# Calculate EMA indicators
data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()
data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()

# Generate trading signals
data["Signal"] = 0

# Buy when EMA 20 crosses above EMA 50
data.loc[
    (data["EMA_20"] > data["EMA_50"]) &
    (data["EMA_20"].shift(1) <= data["EMA_50"].shift(1)),
    "Signal"
] = 1

# Sell when EMA 20 crosses below EMA 50
data.loc[
    (data["EMA_20"] < data["EMA_50"]) &
    (data["EMA_20"].shift(1) >= data["EMA_50"].shift(1)),
    "Signal"
] = -1

# Shift signals to execute on the next trading day's Open
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

# Calculate final portfolio value using last Close
final_value = cash + shares * data["Close"].iloc[-1]

total_return = ((final_value - capital) / capital) * 100

# Calculate buy-and-hold return for comparison
buy_hold_return = (
    (data["Close"].iloc[-1] / data["Close"].iloc[0]) - 1
) * 100

# Display results
print("EMA 20/50 Crossover Strategy")
print("--------------------------------")
print("Initial Capital: ₹", capital)
print("Final Portfolio Value: ₹", round(final_value, 2))
print("Strategy Return: ", round(total_return, 2), "%")
print("Buy and Hold Return: ", round(buy_hold_return, 2), "%")
print("Completed Trades: ", len(trades) // 2)
print("Total Executed Orders: ", len(trades))

print("\nTrade History:")
print(pd.DataFrame(trades))

# Plot price and EMA indicators
plt.figure(figsize=(14, 6))
plt.plot(data.index, data["Close"], label="Close Price")
plt.plot(data.index, data["EMA_20"], label="EMA 20")
plt.plot(data.index, data["EMA_50"], label="EMA 50")

buy_dates = data[data["Signal"] == 1]
sell_dates = data[data["Signal"] == -1]

plt.scatter(
    buy_dates.index, buy_dates["Close"],
    marker="^", label="Buy Signal"
)
plt.scatter(
    sell_dates.index, sell_dates["Close"],
    marker="v", label="Sell Signal"
)

plt.title("RELIANCE - EMA 20/50 Crossover Strategy")
plt.xlabel("Date")
plt.ylabel("Price (₹)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

