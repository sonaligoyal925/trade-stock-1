
# Tradegini Assignment
## Client Churn Prediction and AI-Powered Trading Strategy Analysis

**Candidate:** Sonali Goyal

---

# 1. Project Overview

This project consists of two major parts:

### Part A – Client Churn Prediction

The objective is to identify clients who are likely to stop trading and generate a prioritized list of 200 clients for proactive engagement.

The workflow includes:
- Data cleaning and preprocessing.
- Exploratory data analysis.
- Client-level feature engineering.
- Historical snapshot generation.
- Churn prediction using machine learning.
- Model comparison against a simple baseline.
- Ranking the top 200 high-risk clients.
- Estimating potential brokerage at risk.

### Part B – AI-Powered Trading Strategy Analysis

The objective is to develop a reusable AI prompt that converts plain-English trading ideas into Python trading strategies.

Three strategies were implemented and evaluated:

1. EMA 20/50 Crossover.
2. RSI + High Volume.
3. Daily Range Proxy.

The strategies were compared against a buy-and-hold benchmark using estimated transaction costs and performance metrics.

---

# 2. Project Structure

```text
tradegini_assignment_data/
│
├── clients.csv
├── orders.csv
├── clients_clean.csv
├── orders_clean.csv
│
├── part_a_pipeline.py
├── download_market_data.py
├── strategy_1_ema.py
├── strategy_2_rsi.py
├── strategy_3_daily_range.py
├── compare_strategies.py
│
├── market_data.csv
├── top200.csv
├── model_comparison.csv
├── evaluation_summary.csv
├── feature_importance.csv
├── strategy_comparison.csv
│
├── strategy_prompt.md
├── AI_LOG.md
├── README.md
│
├── Manager Memo.docx
├── Manager Memo.pdf
│
├── requirements.txt
└── .venv/
```

---

# 3. Environment Setup

Python 3 is required.

### Step 1: Create a virtual environment

```bash
python -m venv .venv
```

### Step 2: Activate the environment

Windows:

```bash
.venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

---

# 4. Part A – Churn Prediction Pipeline

## 4.1 Business Objective

Identify the 200 clients most likely to stop trading in September 2026.

Prediction date: 31 August 2026.

Eligible clients must have executed at least one order in July or August 2026.

Churn definition: No executed orders during September 2026.

September information is used only for target generation and evaluation, not predictive features.

## 4.2 Data Preparation

The original datasets contain:

- Client information.
- Order information.
- Trading activity.
- Brokerage and turnover.
- Account attributes.

Data preparation includes:
- Duplicate removal.
- Missing-value handling.
- Standardization of categorical values.
- Validation of client IDs.
- Date and trading-record checks.
- Handling of invalid turnover and brokerage records.

Cleaned dataset sizes:

| Dataset | Records |
|---|---:|
| Clients | 4,000 |
| Orders | 183,635 |

## 4.3 Feature Engineering

Client-level features include:

- Total executed orders.
- Total turnover.
- Active trading days.
- Days since last trade.
- Recent trading activity.
- Previous-period activity.
- Account age.
- City tier.
- Age band.
- Acquisition channel.
- F&O enabled.
- API enabled.

The last login date was excluded to avoid potential information leakage.

## 4.4 Models Implemented

Two machine learning models were evaluated:

- Logistic Regression.
- Random Forest Classifier.

An inactivity-based baseline was also implemented.

### July validation results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| ROC-AUC | 0.8174 | 0.8413 |
| PR-AUC | 0.5935 | 0.6307 |
| Precision@200 | 69% | 75% |
| Churners in Top 200 | 138 | 150 |

Random Forest was selected for final scoring.

## 4.5 September Evaluation

Final eligible cohort: 2,241 clients.

Actual September churners: 614.

| Metric | Random Forest | Baseline |
|---|---:|---:|
| Churners in Top 200 | 145 | 143 |
| Precision@200 | 72.5% | 71.5% |
| Lift@200 | 2.6461 | 2.6096 |

The final Random Forest model identified 145 actual churners in its top 200 ranked clients.

## 4.6 Business Recommendations

The top 200 clients are prioritized for proactive outreach.

Suggested actions include:
- Contact clients with prolonged inactivity.
- Review declining trading activity.
- Conduct relationship calls for other high-risk clients.
- Understand reasons for reduced trading engagement.
- Recommend relevant engagement initiatives based on client needs.

Potential brokerage estimates:

- Historical monthly brokerage: ₹2,629.98.
- Risk-weighted brokerage: ₹2,154.31.

These are estimates and not guaranteed future revenue losses.

## 4.7 Run Part A

Execute:

```bash
python part_a_pipeline.py
```

The pipeline generates:

- top200.csv
- model_comparison.csv
- evaluation_summary.csv
- feature_importance.csv

---

# 5. Part B – AI-Powered Trading Strategy Analysis

## 5.1 Objective

Create a reusable master AI prompt that converts plain-English trading ideas into:

- Strategy explanations.
- Technical indicator calculations.
- Entry and exit rules.
- Python trading code.
- Backtesting results.
- Price charts with buy and sell signals.

The master prompt is available in `strategy_prompt.md`.

## 5.2 Market Data

Historical market data was obtained using yfinance.

| Parameter | Value |
|---|---|
| Instrument | RELIANCE.NS |
| Frequency | Daily |
| Start Date | 1 January 2025 |
| End Date | 30 September 2026 |
| Records | 434 |
| Data Fields | Open, High, Low, Close, Volume |

The data was saved in `market_data.csv`.

## 5.3 Strategy 1 – EMA 20/50 Crossover

Rules:

- Buy when EMA 20 crosses above EMA 50.
- Sell when EMA 20 crosses below EMA 50.

The strategy identifies changes in short-term and long-term price trends.

## 5.4 Strategy 2 – RSI + High Volume

Rules:

- RSI period: 14 days.
- Buy when RSI is below 30.
- Volume must be greater than 1.2 times its 20-day average.
- Sell when RSI crosses above 50.

This strategy combines oversold conditions with higher-than-average trading activity.

## 5.5 Strategy 3 – Daily Range Proxy

Range Position:

```text
(Close - Low) / (High - Low)
```

Rules:

- Buy when Range Position is less than or equal to 0.20.
- Sell when Range Position is greater than or equal to 0.80.

Important limitation:

Daily OHLC data cannot confirm execution at the exact daily low or high. Therefore, this strategy uses a closing-price range proxy rather than claiming exact intraday execution.

## 5.6 Backtesting Assumptions

| Parameter | Value |
|---|---|
| Initial Capital | ₹1,00,000 |
| Brokerage | 0.03% per order |
| Slippage | 0.05% per order |
| Trading Approach | Long-only |
| Leverage | None |
| Maximum Open Positions | 1 |
| Signal Execution | Next trading day's Open |
| Final Valuation | Hypothetical liquidation at final Close |

Brokerage and slippage are illustrative assumptions and not actual broker charges.

## 5.7 Final Strategy Comparison

Results after estimated transaction costs:

| Strategy | Final Portfolio Value | Net Return |
|---|---:|---:|
| EMA 20/50 | ₹90,990.53 | -9.01% |
| RSI + High Volume | ₹1,03,380.16 | +3.38% |
| Daily Range Proxy | ₹84,722.41 | -15.28% |
| Buy and Hold | ₹98,412.36 | -1.59% |

### Risk and trading metrics

| Strategy | Max Drawdown | Completed Trades | Win Rate |
|---|---:|---:|---:|
| EMA 20/50 | -14.88% | 5 | 20.0% |
| RSI + High Volume | -2.21% | 1 | 100.0% |
| Daily Range Proxy | -22.33% | 46 | 58.7% |
| Buy and Hold | -25.20% | 1 | N/A |

### Key observations

- RSI + High Volume generated the highest observed return.
- EMA 20/50 underperformed the benchmark.
- Daily Range Proxy generated the highest transaction costs.
- RSI's 100% win rate is based on only one completed trade.
- More historical data and out-of-sample testing are required before drawing conclusions about strategy reliability.

## 5.8 Run Part B

### Step 1: Download market data

```bash
python download_market_data.py
```

### Step 2: Run individual strategies

```bash
python strategy_1_ema.py
python strategy_2_rsi.py
python strategy_3_daily_range.py
```

### Step 3: Run common comparison

```bash
python compare_strategies.py
```

The comparison script generates:

```text
strategy_comparison.csv
```

---

# 6. Performance Metrics Explained

| Metric | Meaning |
|---|---|
| ROC-AUC | Measures how well a classification model ranks positive and negative outcomes. |
| PR-AUC | Measures precision-recall performance, especially useful for imbalanced data. |
| Precision@200 | Percentage of actual churners among the top 200 predicted clients. |
| Lift@200 | Precision@200 divided by the overall churn rate. |
| Net Return | Portfolio return after estimated transaction costs. |
| Maximum Drawdown | Largest percentage decline from a previous portfolio peak. |
| Win Rate | Percentage of completed trades that were profitable. |
| Profit Factor | Gross profits divided by gross losses. |

---

# 7. Important Limitations

## Part A

- Client behavior may change over time.
- Historical patterns may not continue.
- Feature importance indicates predictive contribution, not causation.
- Brokerage at risk is an estimate.
- Model performance should be monitored on future cohorts.

## Part B

- Only one stock was tested.
- The historical period is relatively limited.
- Transaction costs are illustrative.
- Taxes and other market charges are not included.
- No walk-forward validation has been conducted.
- RSI produced only one completed trade.
- Daily OHLC data cannot establish exact intraday execution at daily extremes.
- Historical performance does not guarantee future returns.

---

# 8. Deliverables Checklist

- [x] Data cleaning and preprocessing.
- [x] Exploratory data analysis.
- [x] Churn prediction pipeline.
- [x] Logistic Regression model.
- [x] Random Forest model.
- [x] Baseline comparison.
- [x] Top 200 client ranking.
- [x] September evaluation.
- [x] Manager memo draft.
- [x] Master AI strategy prompt.
- [x] EMA 20/50 strategy.
- [x] RSI + High Volume strategy.
- [x] Daily Range Proxy strategy.
- [x] Common backtesting framework.
- [x] Transaction cost assumptions.
- [x] Buy-and-hold comparison.
- [x] AI usage log.
- [x] README documentation.

---

# 9. Conclusion

This project demonstrates the use of data analytics, machine learning, and AI-assisted development for two different financial applications.

Part A focuses on identifying potentially inactive trading clients and prioritizing them for business outreach.

Part B demonstrates how natural-language trading ideas can be translated into Python strategies and evaluated using historical market data.

The final results highlight the importance of data quality, validation, transparent assumptions, and careful interpretation of model and backtesting outputs.
