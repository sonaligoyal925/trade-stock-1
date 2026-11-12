
import pandas as pd
import numpy as np

# Load Data and assumptions 

data = pd.read_csv(
    "market_data.csv",
    index_col=0,
    parse_dates=True
)

initial_capital = 100000

brokerage_rate = 0.0003
slippage_rate = 0.0005

# indicators 

# EMA indicators
data["EMA_20"] = data["Close"].ewm(
    span=20, adjust=False
).mean()

data["EMA_50"] = data["Close"].ewm(
    span=50, adjust=False
).mean()


# RSI 14
price_change = data["Close"].diff()

gain = price_change.clip(lower=0)
loss = -price_change.clip(upper=0)

avg_gain = gain.ewm(
    alpha=1/14, min_periods=14, adjust=False
).mean()

avg_loss = loss.ewm(
    alpha=1/14, min_periods=14, adjust=False
).mean()

rs = avg_gain / avg_loss

data["RSI"] = 100 - (100 / (1 + rs))


# Average volume
data["Volume_Avg_20"] = data["Volume"].rolling(20).mean()


# Daily range position
daily_range = data["High"] - data["Low"]

data["Range_Position"] = np.where(
    daily_range > 0,
    (data["Close"] - data["Low"]) / daily_range,
    np.nan
)


# signals 

# EMA crossover
data["EMA_Signal"] = 0

data.loc[
    (data["EMA_20"] > data["EMA_50"]) &
    (data["EMA_20"].shift(1) <= data["EMA_50"].shift(1)),
    "EMA_Signal"
] = 1

data.loc[
    (data["EMA_20"] < data["EMA_50"]) &
    (data["EMA_20"].shift(1) >= data["EMA_50"].shift(1)),
    "EMA_Signal"
] = -1


# RSI + high volume
data["RSI_Signal"] = 0

data.loc[
    (data["RSI"] < 30) &
    (data["Volume"] > 1.2 * data["Volume_Avg_20"]),
    "RSI_Signal"
] = 1

data.loc[
    (data["RSI"] > 50) &
    (data["RSI"].shift(1) <= 50),
    "RSI_Signal"
] = -1


# Daily range proxy
data["Range_Signal"] = 0

data.loc[
    data["Range_Position"] <= 0.20,
    "Range_Signal"
] = 1

data.loc[
    data["Range_Position"] >= 0.80,
    "Range_Signal"
] = -1


# common backtesting function

def backtest(data, signal_column):

    cash = initial_capital
    shares = 0

    trades = []
    portfolio_values = []

    # Execute each signal on the next trading day's Open
    signals = data[signal_column].shift(1).fillna(0)

    for i in range(len(data)):

        price = data["Open"].iloc[i]
        signal = signals.iloc[i]
        date = data.index[i]

        # Buy
        if signal == 1 and shares == 0:

            buy_price = price * (1 + slippage_rate)

            cost_per_share = (
                buy_price * (1 + brokerage_rate)
            )

            shares = int(cash // cost_per_share)

            if shares > 0:

                trade_value = shares * buy_price
                brokerage = trade_value * brokerage_rate

                cash -= trade_value + brokerage

                trades.append({
                    "Date": date,
                    "Action": "BUY",
                    "Price": buy_price,
                    "Shares": shares,
                    "Brokerage": brokerage
                })

        # Sell
        elif signal == -1 and shares > 0:

            sell_price = price * (1 - slippage_rate)

            trade_value = shares * sell_price
            brokerage = trade_value * brokerage_rate

            cash += trade_value - brokerage

            trades.append({
                "Date": date,
                "Action": "SELL",
                "Price": sell_price,
                "Shares": shares,
                "Brokerage": brokerage
            })

            shares = 0

        # Daily marked to market value
        portfolio_value = (
            cash + shares * data["Close"].iloc[i]
        )

        portfolio_values.append(portfolio_value)

    # FINAL LIQUIDATION

    final_close = data["Close"].iloc[-1]

    if shares > 0:

        final_sell_price = (
            final_close * (1 - slippage_rate)
        )

        final_sale_value = shares * final_sell_price

        final_exit_brokerage = (
            final_sale_value * brokerage_rate
        )

        final_value = (
            cash + final_sale_value - final_exit_brokerage
        )

        open_position = "Yes"

        total_brokerage = (
            sum(t["Brokerage"] for t in trades)
            + final_exit_brokerage
        )

    else:

        final_value = cash
        open_position = "No"

        total_brokerage = sum(
            t["Brokerage"] for t in trades
        )

    # performance metrics

    total_return = (
        (final_value / initial_capital) - 1
    ) * 100

    # Maximum drawdown based on daily closing values
    portfolio_series = pd.Series(
        portfolio_values,
        index=data.index
    )

    running_peak = portfolio_series.cummax()

    drawdown = (
        portfolio_series - running_peak
    ) / running_peak

    max_drawdown = drawdown.min() * 100

    # Completed trade profits
    trade_profits = []

    buy_trade = None

    for trade in trades:

        if trade["Action"] == "BUY":

            buy_trade = trade

        elif trade["Action"] == "SELL" and buy_trade is not None:

            buy_cost = (
                buy_trade["Price"] * buy_trade["Shares"]
                + buy_trade["Brokerage"]
            )

            sell_proceeds = (
                trade["Price"] * trade["Shares"]
                - trade["Brokerage"]
            )

            trade_profits.append(
                sell_proceeds - buy_cost
            )

            buy_trade = None

    # Win rate and profit factor
    if trade_profits:

        wins = [
            p for p in trade_profits if p > 0
        ]

        losses = [
            p for p in trade_profits if p < 0
        ]

        win_rate = (
            len(wins) / len(trade_profits)
        ) * 100

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))

        if gross_loss > 0:

            profit_factor = gross_profit / gross_loss

        elif gross_profit > 0:

            profit_factor = np.inf

        else:

            profit_factor = 0

    else:

        win_rate = 0
        profit_factor = 0

    return {
        "Final Value": round(final_value, 2),
        "Net Return (%)": round(total_return, 2),
        "Max Drawdown (%)": round(max_drawdown, 2),
        "Completed Trades": len(trade_profits),
        "Open Position": open_position,
        "Win Rate (%)": round(win_rate, 2),
        "Profit Factor": round(profit_factor, 2),
        "Total Brokerage": round(total_brokerage, 2)
    }


# buy and hold benchmark 

def buy_and_hold(data):

    # Buy at first trading day's Open
    buy_price = (
        data["Open"].iloc[0] * (1 + slippage_rate)
    )

    cost_per_share = (
        buy_price * (1 + brokerage_rate)
    )

    shares = int(initial_capital // cost_per_share)

    entry_value = shares * buy_price
    entry_brokerage = entry_value * brokerage_rate

    cash_remaining = (
        initial_capital - entry_value - entry_brokerage
    )

    # Daily closing valuation before hypothetical exit costs
    daily_values = (
        cash_remaining + shares * data["Close"]
    )

    # Hypothetical liquidation at final Close
    sell_price = (
        data["Close"].iloc[-1] * (1 - slippage_rate)
    )

    exit_value = shares * sell_price
    exit_brokerage = exit_value * brokerage_rate

    final_value = (
        cash_remaining + exit_value - exit_brokerage
    )

    total_return = (
        (final_value / initial_capital) - 1
    ) * 100

    # Drawdown based on daily closing valuations
    running_peak = daily_values.cummax()

    drawdown = (
        daily_values - running_peak
    ) / running_peak

    max_drawdown = drawdown.min() * 100

    return {
        "Final Value": round(final_value, 2),
        "Net Return (%)": round(total_return, 2),
        "Max Drawdown (%)": round(max_drawdown, 2),
        "Completed Trades": 1,
        "Open Position": "No",
        "Win Rate (%)": "N/A",
        "Profit Factor": "N/A",
        "Total Brokerage": round(
            entry_brokerage + exit_brokerage, 2
        )
    }

results = {}

results["EMA 20/50"] = backtest(
    data, "EMA_Signal"
)

results["RSI + High Volume"] = backtest(
    data, "RSI_Signal"
)

results["Daily Range Proxy"] = backtest(
    data, "Range_Signal"
)

results["Buy and Hold"] = buy_and_hold(data)


comparison = pd.DataFrame(results).T

print("\nSTRATEGY PERFORMANCE COMPARISON")
print("--------------------------------")

print(comparison.to_string())

comparison.to_csv("strategy_comparison.csv")

print("\nComparison saved to strategy_comparison.csv")

print("\nTRANSACTION COST ASSUMPTIONS")
print("--------------------------------")
print("Brokerage per order:", brokerage_rate * 100, "%")
print("Slippage per order:", slippage_rate * 100, "%")
print("Initial Capital: ₹", initial_capital)

print("\nAll strategies use hypothetical final liquidation.")
print("Daily drawdown uses closing-price portfolio valuation.")
