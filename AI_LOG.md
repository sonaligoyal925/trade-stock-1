
# AI Usage Log – Tradegini Assignment

**Project:** Tradegini – Churn Prediction and AI-Powered Trading Strategy Analysis  
**Candidate:** Sonali Goyal  
**Tools Used:** ChatGPT, Python, Pandas, NumPy, Scikit-learn, Matplotlib, yfinance  
**Environment:** Visual Studio Code, Python Virtual Environment  

---

# 1. Purpose of AI Assistance

AI tools were used as development, research, and learning assistants throughout this assignment.

The primary objective was to use AI to accelerate development while maintaining human control over data validation, analytical decisions, code execution, and interpretation.

AI assistance was used for:

- Understanding the assignment requirements.
- Breaking complex tasks into smaller implementation steps.
- Generating initial Python code drafts.
- Explaining statistical and machine learning concepts.
- Identifying potential data leakage and implementation issues.
- Improving code structure and readability.
- Interpreting model and backtesting results.
- Preparing business-oriented documentation.

All generated code was reviewed and executed locally. Outputs were checked before being included in the assignment.

---

# 2. Part A – Client Churn Prediction

## 2.1 Data Understanding and Cleaning

AI assistance was used to understand the client and order datasets and structure the data preparation workflow.

Areas where AI helped:

- Understanding the data dictionary and dataset relationships.
- Identifying missing values and duplicate records.
- Reviewing inconsistent categorical values.
- Identifying invalid turnover and brokerage records.
- Handling unknown client IDs.
- Understanding date-related data quality issues.
- Developing a structured data cleaning process.

Human verification:

- Loaded and inspected the original CSV files.
- Reviewed dataset dimensions and column types.
- Checked duplicate records and missing values.
- Reviewed the impact of cleaning decisions.
- Executed the cleaning process locally.
- Verified the resulting cleaned datasets.

The final cleaned datasets contained:

- 4,000 clients.
- 183,635 order records.

## 2.2 Churn Definition and Feature Engineering

AI assistance helped translate the business problem into a machine learning problem.

The following business definitions were implemented:

- Prediction date: 31 August 2026.
- Eligible clients: Clients with at least one executed order in July or August 2026.
- Churn definition: No executed orders during September 2026.
- September data was used only for target generation and evaluation, not predictive features.

AI assistance was used for developing client-level features such as:

- Total executed orders.
- Total turnover.
- Number of active trading days.
- Days since last trade.
- Recent trading activity.
- Previous-period trading activity.
- Account age.
- Client demographic and account attributes.

Human verification:

- Checked the prediction-date cutoff.
- Reviewed the eligible client cohort.
- Excluded September information from predictive features.
- Excluded last_login_date to avoid potential leakage.
- Validated historical monthly snapshots.

## 2.3 Model Development

AI assistance was used to develop and understand the machine learning workflow.

Models implemented:

1. Logistic Regression.
2. Random Forest Classifier.

AI assistance covered:

- Numerical and categorical preprocessing.
- Missing-value imputation.
- Feature scaling.
- One-hot encoding.
- Class imbalance handling.
- Model training and prediction.
- Evaluation metric calculations.
- Feature importance interpretation.

The models were compared against a simple inactivity-based baseline.

## 2.4 Model Validation Results

Validation was performed using the July historical snapshot.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| ROC-AUC | 0.8174 | 0.8413 |
| PR-AUC | 0.5935 | 0.6307 |
| Precision@200 | 69% | 75% |
| Churners in Top 200 | 138 | 150 |

Baseline performance:

- Top 200 churners identified: 140.
- Precision@200: 70%.
- Lift@200: 2.57.

Random Forest performed better than Logistic Regression on the primary ranking metrics and was selected for final scoring.

## 2.5 Final Scoring and Evaluation

The final model was trained using historical snapshots from April to July 2026 and applied to the August 2026 eligible cohort.

Final cohort:

- Eligible clients: 2,241.
- Actual September churners: 614.
- September churn rate: 27.40%.

Final evaluation:

| Metric | Random Forest | Inactivity Baseline |
|---|---:|---:|
| Churners in Top 200 | 145 | 143 |
| Precision@200 | 72.5% | 71.5% |
| Lift@200 | 2.6461 | 2.6096 |

Additional analysis:

- Feature importance was reviewed to understand the main predictive factors.
- Days since last trade was the most influential feature.
- Top 200 clients were ranked by model risk score.
- Suggested call actions were generated using inactivity and trading activity patterns.
- Potential brokerage at risk was estimated.

Final brokerage estimates:

- Historical monthly brokerage: ₹2,629.98.
- Risk-weighted brokerage: ₹2,154.31.

These estimates are analytical indicators and should not be interpreted as guaranteed future revenue loss.

---

# 3. Part B – AI-Powered Trading Strategy Analysis

## 3.1 Master AI Prompt Development

AI assistance was used to design a reusable master prompt that converts plain-English trading ideas into Python trading strategies.

The prompt was designed to request:

- Explanation of the trading idea.
- Indicator calculations.
- Entry and exit conditions.
- Python implementation using Pandas and NumPy.
- Backtesting assumptions.
- Buy and sell signals.
- Portfolio performance calculations.
- Price charts with trading markers.
- Discussion of assumptions and limitations.

The objective was to make the strategy-generation process reusable for different trading ideas.

## 3.2 Market Data Collection

Since the assignment's client and order datasets did not contain historical market OHLCV prices, external market data was obtained using yfinance.

Dataset details:

- Instrument: RELIANCE.NS.
- Data source: Yahoo Finance through yfinance.
- Frequency: Daily.
- Start date: 1 January 2025.
- End date: 30 September 2026.
- Records: 434.
- Columns: Open, High, Low, Close and Volume.
- Missing values: None.

The downloaded data was saved as market_data.csv.

The data was adjusted using yfinance's auto_adjust=True setting.

## 3.3 Strategy 1 – EMA 20/50 Crossover

AI assistance was used to implement an exponential moving average crossover strategy.

Trading rules:

- Buy when EMA 20 crosses above EMA 50.
- Sell when EMA 20 crosses below EMA 50.

Implementation:

- EMA 20 and EMA 50 were calculated using closing prices.
- Crossover signals were generated using current and previous values.
- Signals were executed at the next trading day's Open.
- A long-only approach was used.
- Only one position could be open at a time.

Initial backtest result before transaction costs:

- Initial capital: ₹1,00,000.
- Final portfolio value: ₹91,824.80.
- Return: -8.18%.
- Completed trades: 5.

## 3.4 Strategy 2 – RSI + High Volume

AI assistance was used to implement a momentum-based strategy combining RSI and trading volume.

Trading rules:

- Calculate RSI using a 14-day period.
- Buy when RSI is below 30.
- Require volume to be greater than 1.2 times its 20-day average.
- Sell when RSI crosses above 50.

Implementation:

- RSI was calculated using exponentially smoothed average gains and losses.
- A 20-day average volume was calculated.
- Buy and sell signals were generated from the defined conditions.
- Signals were executed at the next trading day's Open.

Initial backtest result before transaction costs:

- Initial capital: ₹1,00,000.
- Final portfolio value: ₹1,03,707.18.
- Return: +3.71%.
- Completed trades: 1.
- One position remained open at the end of the dataset.

## 3.5 Strategy 3 – Daily Range Proxy

AI assistance was used to translate the daily low/high trading idea into a testable strategy using daily OHLC data.

A limitation was identified: daily OHLC data cannot establish whether an actual trade could execute at the exact daily low and subsequently sell at the exact daily high.

Therefore, a closing-price range proxy was implemented.

Range Position formula:

Range Position = (Close - Low) / (High - Low)

Trading rules:

- Buy when Range Position is less than or equal to 0.20.
- Sell when Range Position is greater than or equal to 0.80.
- Execute signals at the next trading day's Open.

Initial backtest result before transaction costs:

- Initial capital: ₹1,00,000.
- Final portfolio value: ₹91,355.10.
- Return: -8.64%.
- Completed trades: 46.
- One position remained open at the end of the dataset.

This strategy is a daily range proxy and does not represent guaranteed execution at intraday extremes.

---

# 4. Common Backtesting Framework

AI assistance was used to combine the three individual strategies into a common backtesting framework.

The common framework was designed to ensure consistent assumptions across strategies.

## 4.1 Backtesting Assumptions

| Parameter | Assumption |
|---|---|
| Initial Capital | ₹1,00,000 |
| Brokerage | 0.03% per executed order |
| Slippage | 0.05% per executed order |
| Trading Approach | Long-only |
| Leverage | None |
| Open Positions | Maximum one |
| Entry Execution | Next trading day's Open |
| Exit Execution | Next trading day's Open |
| Final Valuation | Hypothetical liquidation at final Close |

Brokerage and slippage are illustrative assumptions, not actual Tradejini brokerage charges.

## 4.2 Performance Metrics

The common framework calculates:

- Final portfolio value.
- Net return after estimated transaction costs.
- Maximum drawdown.
- Number of completed trades.
- Open position status.
- Win rate.
- Profit factor.
- Total estimated brokerage.

The framework also includes a buy-and-hold benchmark using the same transaction cost assumptions.

Open positions are hypothetically liquidated at the final closing price for final portfolio valuation.

Maximum drawdown is calculated using daily closing-price portfolio valuations before hypothetical final liquidation costs.

---

# 5. Final Part B Results

Final results after estimated brokerage and slippage:

| Strategy | Final Value | Net Return |
|---|---:|---:|
| EMA 20/50 | ₹90,990.53 | -9.01% |
| RSI + High Volume | ₹1,03,380.16 | +3.38% |
| Daily Range Proxy | ₹84,722.41 | -15.28% |
| Buy and Hold | ₹98,412.36 | -1.59% |

Additional metrics:

| Strategy | Max Drawdown | Completed Trades | Win Rate |
|---|---:|---:|---:|
| EMA 20/50 | -14.88% | 5 | 20.0% |
| RSI + High Volume | -2.21% | 1 | 100.0% |
| Daily Range Proxy | -22.33% | 46 | 58.7% |
| Buy and Hold | -25.20% | 1 | N/A |

Transaction costs:

| Strategy | Total Estimated Brokerage |
|---|---:|
| EMA 20/50 | ₹283.54 |
| RSI + High Volume | ₹122.63 |
| Daily Range Proxy | ₹2,638.79 |
| Buy and Hold | ₹58.83 |

## 5.1 Key Observations

1. RSI + High Volume produced the highest observed return of 3.38%.
2. EMA 20/50 underperformed the buy-and-hold benchmark.
3. Daily Range Proxy generated the highest trading activity and brokerage costs.
4. Daily Range Proxy had a positive win rate of 58.7%, but its profit factor was only 0.70, indicating that gross losses exceeded gross profits.
5. RSI + High Volume had only one completed trade, so its 100% win rate should not be interpreted as evidence of consistent profitability.
6. RSI + High Volume outperformed buy-and-hold by 4.97 percentage points in this sample.

These are historical sample results and do not guarantee future performance.

---

# 6. AI-Assisted Debugging and Improvements

During development, AI assistance was used to troubleshoot and improve code.

Examples include:

- Correcting historical snapshot eligibility logic in Part A.
- Identifying and resolving a missing churn column issue during baseline evaluation.
- Improving the common backtesting framework.
- Adding estimated transaction costs.
- Adding maximum drawdown, win rate and profit factor.
- Correcting final portfolio valuation for open positions.
- Standardizing the buy-and-hold benchmark assumptions.
- Improving code readability and documentation.

All corrections were tested by executing the Python scripts locally and reviewing their outputs.

---

# 7. Limitations and Areas for Improvement

## Part A – Churn Prediction

- The model is based on historical client activity.
- Client behavior may change over time.
- Predictive feature importance does not establish causation.
- Brokerage at risk is an estimate, not guaranteed revenue loss.
- Model performance should be monitored on future monthly cohorts.

## Part B – Trading Strategies

- Only one instrument, RELIANCE.NS, was tested.
- The evaluation period is relatively limited.
- Brokerage and slippage assumptions are illustrative.
- Taxes and other market charges are not included.
- No out-of-sample or walk-forward validation has been conducted.
- RSI has only one completed trade.
- Daily OHLC data cannot verify exact intraday high/low execution.
- Strategy performance may differ across stocks and market conditions.

Future improvements could include:

- Testing multiple stocks and market conditions.
- Using longer historical datasets.
- Conducting walk-forward validation.
- Performing parameter sensitivity analysis.
- Adding taxes and other transaction charges.
- Using intraday data for strategies requiring precise price execution.
- Evaluating risk-adjusted performance metrics such as Sharpe ratio.

---

# 8. Human Review and Responsibility

AI-generated code and explanations were treated as initial drafts and learning support.

The author:

- Reviewed the assignment requirements.
- Inspected the source datasets.
- Executed the Python scripts locally.
- Checked the generated outputs.
- Reviewed the model evaluation results.
- Verified the backtesting assumptions.
- Interpreted the results.
- Documented limitations and potential improvements.

Final analytical decisions and conclusions were made after reviewing the actual execution results.

AI was used to support the development process, not as a replacement for data validation, independent analysis or human judgment.
