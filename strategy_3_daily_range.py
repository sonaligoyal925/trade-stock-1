
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load market data
data = pd.read_csv("market_data.csv", index_col=0, parse_dates=True)

# Calculate where the closing price lies within the daily range
daily_range = data["High"] - data["Low"]

data["Range_Position"] = np.where(
    daily_range > 0,
    (data["Close"] - data["Low"]) / daily_range,
    np.nan
)

# Generate trading signals
data["Signal"] = 0

# Buy when close is near the daily low
data.loc[data["Range_Position"] <= 0.20, "Signal"] = 1

# Sell when close is near the daily high
data.loc[data["Range_Position"] >= 0.80, "Signal"] = -1

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
print("Daily Range Strategy")
print("--------------------------------")
print("Initial Capital: ₹", capital)
print("Final Portfolio Value: ₹", round(final_value, 2))
print("Strategy Return:", round(total_return, 2), "%")
print("Buy and Hold Return:", round(buy_hold_return, 2), "%")
print("Completed Trades:", len(trades) // 2)
print("Total Executed Orders:", len(trades))

print("\nTrade History:")
print(pd.DataFrame(trades))

# Plot price and trading signals
plt.figure(figsize=(14, 6))

plt.plot(data.index, data["Close"], label="Close Price")

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

plt.title("RELIANCE - Daily Range Strategy")
plt.xlabel("Date")
plt.ylabel("Price (₹)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
