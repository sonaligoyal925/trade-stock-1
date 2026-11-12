
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


#  load and clean data

print("\nLoading datasets...")

clients = pd.read_csv("clients.csv")
orders = pd.read_csv("orders.csv")

clients["onboarding_date"] = pd.to_datetime(
    clients["onboarding_date"]
)
clients["last_login_date"] = pd.to_datetime(
    clients["last_login_date"]
)
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"]
)

clients_clean = clients.copy()
orders_clean = orders.copy()

# Remove exact duplicates
clients_clean = clients_clean.drop_duplicates()
orders_clean = orders_clean.drop_duplicates()

# Handle missing values
clients_clean["age_band"] = (
    clients_clean["age_band"].fillna("Unknown")
)
orders_clean["platform"] = (
    orders_clean["platform"].fillna("Unknown")
)

# Standardize categorical values
for col in ["segment", "platform", "side", "status"]:
    orders_clean[col] = (
        orders_clean[col].str.strip().str.upper()
    )

for col in ["city_tier", "age_band", "acquisition_channel"]:
    clients_clean[col] = (
        clients_clean[col].str.strip().str.upper()
    )

# Remove orders after September 30
orders_clean = orders_clean[
    orders_clean["order_ts"] <= "2026-09-30 23:59:59"
]

# Exclude weekend orders
orders_clean = orders_clean[
    orders_clean["order_ts"].dt.dayofweek < 5
]

# Keep orders belonging to known clients
orders_clean = orders_clean[
    orders_clean["client_id"].isin(
        clients_clean["client_id"]
    )
].copy()

# Flag negative turnover
orders_clean["invalid_turnover"] = (
    orders_clean["turnover_inr"] < 0
)

# Calculate validated brokerage separately
expected_brokerage = np.minimum(
    20,
    orders_clean["turnover_inr"] * 0.0005
)

orders_clean["validated_brokerage"] = np.where(
    orders_clean["status"] == "EXECUTED",
    np.where(
        orders_clean["turnover_inr"] >= 0,
        expected_brokerage,
        np.nan
    ),
    0
)

print("Data loaded and cleaned.")
print("Clients:", clients_clean.shape)
print("Orders:", orders_clean.shape)


# 2. Create historical datesets 

def create_monthly_dataset(orders, clients, prediction_date):

    date = pd.Timestamp(prediction_date)

    # Current month + previous calendar month
    start_date = (
        date.to_period("M") - 1
    ).start_time

    past = orders[
        (orders["order_ts"] <= (
            date + pd.Timedelta(days=1)
            - pd.Timedelta(seconds=1)
        )) &
        (orders["status"] == "EXECUTED")
    ].copy()

    recent = past[
        past["order_ts"] >= start_date
    ]

    eligible = recent["client_id"].unique()

    features = past[
        past["client_id"].isin(eligible)
    ].groupby("client_id").agg(
        total_orders=("order_id", "count"),

        total_turnover=(
            "turnover_inr",
            lambda x: x[x >= 0].sum()
        ),

        active_days=(
            "order_ts",
            lambda x: x.dt.date.nunique()
        ),

        last_trade=("order_ts", "max")
    ).reset_index()

    features["days_since_last_trade"] = (
        date - features["last_trade"].dt.normalize()
    ).dt.days

    features["recent_orders"] = (
        features["client_id"]
        .map(recent.groupby("client_id").size())
        .fillna(0)
    )

    features = features.merge(
        clients.drop(columns=["last_login_date"]),
        on="client_id",
        how="left"
    )

    features["account_age_days"] = (
        date - features["onboarding_date"]
    ).dt.days

    # Next calendar month for churn target
    next_month_start = (
        date.to_period("M") + 1
    ).start_time

    next_month_end = (
        date.to_period("M") + 2
    ).start_time

    future = orders[
        (orders["order_ts"] >= next_month_start) &
        (orders["order_ts"] < next_month_end) &
        (orders["status"] == "EXECUTED")
    ]

    active_next = future["client_id"].unique()

    features["churn"] = (
        ~features["client_id"].isin(active_next)
    ).astype(int)

    return features


print("\nCreating historical datasets...")

april_data = create_monthly_dataset(
    orders_clean, clients_clean, "2026-04-30"
)

may_data = create_monthly_dataset(
    orders_clean, clients_clean, "2026-05-31"
)

june_data = create_monthly_dataset(
    orders_clean, clients_clean, "2026-06-30"
)

july_data = create_monthly_dataset(
    orders_clean, clients_clean, "2026-07-31"
)

train_data = pd.concat(
    [april_data, may_data, june_data],
    ignore_index=True
)

final_train_data = pd.concat(
    [train_data, july_data],
    ignore_index=True
)

print("April:", april_data.shape)
print("May:", may_data.shape)
print("June:", june_data.shape)
print("July:", july_data.shape)
print("Initial training:", train_data.shape)
print("Final training:", final_train_data.shape)


# feature preprocessing and model training

numeric_features = [
    "total_orders",
    "total_turnover",
    "active_days",
    "days_since_last_trade",
    "account_age_days"
]

categorical_features = [
    "city_tier",
    "age_band",
    "acquisition_channel",
    "fno_enabled",
    "api_enabled"
]

feature_columns = (
    numeric_features + categorical_features
)

X_train = train_data[feature_columns]
y_train = train_data["churn"]

X_val = july_data[feature_columns]
y_val = july_data["churn"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


print("\nTraining Logistic Regression...")

logistic_model = Pipeline([
    ("preprocessor", preprocessor),

    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

logistic_model.fit(X_train, y_train)

logistic_probabilities = (
    logistic_model.predict_proba(X_val)[:, 1]
)

logistic_results = july_data[
    ["client_id", "churn"]
].copy()

logistic_results["risk_score"] = (
    logistic_probabilities
)

logistic_top200 = logistic_results.nlargest(
    200, "risk_score"
)

logistic_precision = (
    logistic_top200["churn"].mean()
)


print("Training Random Forest...")

rf_model = Pipeline([
    ("preprocessor", preprocessor),

    ("classifier", RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)

rf_probabilities = (
    rf_model.predict_proba(X_val)[:, 1]
)

rf_results = july_data[
    ["client_id", "churn"]
].copy()

rf_results["risk_score"] = rf_probabilities

rf_top200 = rf_results.nlargest(
    200, "risk_score"
)

rf_precision = rf_top200["churn"].mean()


#  july baseline and model comparison

baseline_top200 = july_data.nlargest(
    200, "days_since_last_trade"
)

baseline_precision = (
    baseline_top200["churn"].mean()
)

overall_validation_churn = y_val.mean()

model_comparison = pd.DataFrame({
    "Model": [
        "Inactivity Baseline",
        "Logistic Regression",
        "Random Forest"
    ],

    "ROC_AUC": [
        np.nan,
        roc_auc_score(y_val, logistic_probabilities),
        roc_auc_score(y_val, rf_probabilities)
    ],

    "PR_AUC": [
        np.nan,
        average_precision_score(y_val, logistic_probabilities),
        average_precision_score(y_val, rf_probabilities)
    ],

    "Precision_at_200": [
        baseline_precision,
        logistic_precision,
        rf_precision
    ],

    "Lift_at_200": [
        baseline_precision / overall_validation_churn,
        logistic_precision / overall_validation_churn,
        rf_precision / overall_validation_churn
    ]
})

model_comparison.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nJuly Validation Results")
print("-----------------------")
print(model_comparison.round(4).to_string(index=False))


# train final random forest model 

print("\nTraining final Random Forest model...")

X_final_train = final_train_data[feature_columns]
y_final_train = final_train_data["churn"]

final_rf_model = Pipeline([
    ("preprocessor", preprocessor),

    ("classifier", RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

final_rf_model.fit(
    X_final_train,
    y_final_train
)


# create august 31 client features

print("\nCreating August 31 client features...")

prediction_date = pd.Timestamp("2026-08-31")

orders_before_sep = orders_clean[
    (orders_clean["order_ts"] <= "2026-08-31 23:59:59") &
    (orders_clean["status"] == "EXECUTED")
].copy()

# Eligible clients: at least one executed order
# in July or August

recent_clients = orders_before_sep[
    orders_before_sep["order_ts"].dt.month.isin([7, 8])
]["client_id"].unique()

client_features = orders_before_sep.groupby(
    "client_id"
).agg(
    total_orders=("order_id", "count"),

    total_turnover=(
        "turnover_inr",
        lambda x: x[x >= 0].sum()
    ),

    active_days=(
        "order_ts",
        lambda x: x.dt.date.nunique()
    ),

    last_trade=("order_ts", "max")
).reset_index()

client_features["days_since_last_trade"] = (
    prediction_date -
    client_features["last_trade"].dt.normalize()
).dt.days


# recent trading activity 

last_30 = orders_before_sep[
    orders_before_sep["order_ts"] >= "2026-08-02"
]

previous_30 = orders_before_sep[
    (orders_before_sep["order_ts"] >= "2026-07-03") &
    (orders_before_sep["order_ts"] < "2026-08-02")
]

recent_activity = last_30.groupby("client_id").size()
previous_activity = previous_30.groupby("client_id").size()

client_features["orders_last_30d"] = (
    client_features["client_id"]
    .map(recent_activity)
    .fillna(0)
)

client_features["orders_previous_30d"] = (
    client_features["client_id"]
    .map(previous_activity)
    .fillna(0)
)


# merge client info 

client_features = client_features.merge(
    clients_clean.drop(columns=["last_login_date"]),
    on="client_id",
    how="left"
)

client_features["account_age_days"] = (
    prediction_date -
    client_features["onboarding_date"]
).dt.days

# Apply eligibility rule
client_features = client_features[
    client_features["client_id"].isin(recent_clients)
].copy()

print("Eligible August clients:", len(client_features))


# final churn scores

X_september = client_features[feature_columns]

september_probabilities = (
    final_rf_model.predict_proba(X_september)[:, 1]
)

final_scores = client_features[
    ["client_id"]
].copy()

final_scores["risk_score"] = (
    september_probabilities
)

final_top200 = final_scores.nlargest(
    200, "risk_score"
).reset_index(drop=True)

final_top200["rank"] = (
    final_top200.index + 1
)


# top 200 calling list

call_list = final_top200.merge(
    client_features[
        [
            "client_id",
            "days_since_last_trade",
            "orders_last_30d",
            "orders_previous_30d",
            "total_orders",
            "total_turnover",
            "city_tier",
            "acquisition_channel"
        ]
    ],
    on="client_id",
    how="left"
)

conditions = [
    call_list["days_since_last_trade"] >= 30,

    call_list["orders_last_30d"] <
    call_list["orders_previous_30d"]
]

actions = [
    "Reactivation call - prolonged inactivity",

    "Engagement call - declining trading activity"
]

call_list["suggested_action"] = np.select(
    conditions,
    actions,
    default="Relationship call - understand trading needs"
)


# brokage exposure 

print("\nCalculating brokerage exposure...")

recent_brokerage = orders_clean[
    (orders_clean["order_ts"] >= "2026-07-01") &
    (orders_clean["order_ts"] < "2026-09-01") &
    (orders_clean["status"] == "EXECUTED")
].copy()

brokerage_by_client = (
    recent_brokerage.groupby("client_id")[
        "validated_brokerage"
    ]
    .sum()
    .reset_index(name="brokerage_2_months")
)

brokerage_by_client["avg_monthly_brokerage"] = (
    brokerage_by_client["brokerage_2_months"] / 2
)

call_list = call_list.merge(
    brokerage_by_client[
        ["client_id", "avg_monthly_brokerage"]
    ],
    on="client_id",
    how="left"
)

call_list["avg_monthly_brokerage"] = (
    call_list["avg_monthly_brokerage"].fillna(0)
)

total_brokerage_at_risk = (
    call_list["avg_monthly_brokerage"].sum()
)

probability_weighted_brokerage = (
    call_list["avg_monthly_brokerage"] *
    call_list["risk_score"]
).sum()


#  september actual outcomes

print("\nEvaluating September outcomes...")

september_active = orders_clean[
    (orders_clean["order_ts"].dt.month == 9) &
    (orders_clean["status"] == "EXECUTED")
]["client_id"].unique()

# Actual outcomes are created separately.
# They are used only for evaluation, never for prediction.

actual_outcomes = client_features[
    ["client_id"]
].copy()

actual_outcomes["churn"] = (
    ~actual_outcomes["client_id"].isin(september_active)
).astype(int)


# evaluate random forest 

rf_evaluation = call_list.merge(
    actual_outcomes,
    on="client_id",
    how="left"
)

rf_churners = rf_evaluation["churn"].sum()

rf_precision_final = (
    rf_churners / 200
)

total_churners = (
    actual_outcomes["churn"].sum()
)

overall_churn_rate = (
    actual_outcomes["churn"].mean()
)

rf_recall = (
    rf_churners / total_churners
)

rf_lift = (
    rf_precision_final / overall_churn_rate
)


# evaluate baseline on same cohort

# IMPORTANT FIX:
# client_features does not contain the churn column.
# Therefore, merge actual_outcomes before calculating
# baseline churners.

final_baseline = client_features.nlargest(
    200,
    "days_since_last_trade"
).merge(
    actual_outcomes,
    on="client_id",
    how="left"
)

baseline_churners = (
    final_baseline["churn"].sum()
)

baseline_precision_final = (
    baseline_churners / 200
)

baseline_lift_final = (
    baseline_precision_final / overall_churn_rate
)


# feature importance 

feature_names = (
    final_rf_model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

importances = (
    final_rf_model
    .named_steps["classifier"]
    .feature_importances_
)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})


def get_original_feature(name):

    name = (
        name.replace("num__", "")
        .replace("cat__", "")
    )

    for col in categorical_features:
        if name.startswith(col + "_"):
            return col

    return name


importance_df["original_feature"] = (
    importance_df["feature"]
    .apply(get_original_feature)
)

grouped_importance = (
    importance_df.groupby("original_feature")["importance"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

grouped_importance.to_csv(
    "feature_importance.csv",
    index=False
)


# top 200 client export

final_columns = [
    "rank",
    "client_id",
    "risk_score",
    "days_since_last_trade",
    "orders_last_30d",
    "orders_previous_30d",
    "total_orders",
    "total_turnover",
    "city_tier",
    "acquisition_channel",
    "avg_monthly_brokerage",
    "suggested_action"
]

top200_export = call_list[
    final_columns
].copy()

top200_export.to_csv(
    "top200.csv",
    index=False
)


# summary

evaluation_summary = pd.DataFrame({
    "Metric": [
        "Eligible clients",
        "Actual churners",
        "Overall churn rate",
        "Random Forest top-200 churners",
        "Random Forest Precision@200",
        "Random Forest Recall@200",
        "Random Forest Lift@200",
        "Baseline top-200 churners",
        "Baseline Precision@200",
        "Baseline Lift@200",
        "Historical monthly brokerage at risk",
        "Risk-weighted brokerage estimate"
    ],

    "Value": [
        len(actual_outcomes),
        total_churners,
        overall_churn_rate,
        rf_churners,
        rf_precision_final,
        rf_recall,
        rf_lift,
        baseline_churners,
        baseline_precision_final,
        baseline_lift_final,
        total_brokerage_at_risk,
        probability_weighted_brokerage
    ]
})

evaluation_summary.to_csv(
    "evaluation_summary.csv",
    index=False
)


print("\n")
print("=" * 50)
print("TRADEJINI CHURN PIPELINE COMPLETED")
print("=" * 50)

print("\nFINAL MODEL SUMMARY")
print("-------------------")
print("Historical training rows:", len(final_train_data))
print("Eligible August clients:", len(client_features))
print("Selected clients:", len(top200_export))

print("\nSEPTEMBER EVALUATION")
print("--------------------")
print("Actual churners:", total_churners)

print("\nRandom Forest:")
print("Top-200 churners:", rf_churners)
print("Precision@200:", round(rf_precision_final, 4))
print("Recall@200:", round(rf_recall, 4))
print("Lift@200:", round(rf_lift, 4))

print("\nInactivity Baseline:")
print("Top-200 churners:", baseline_churners)
print("Precision@200:", round(baseline_precision_final, 4))
print("Lift@200:", round(baseline_lift_final, 4))

print("\nBROKERAGE IMPACT")
print("----------------")
print(
    "Historical monthly brokerage:",
    round(total_brokerage_at_risk, 2)
)

print(
    "Risk-weighted estimate:",
    round(probability_weighted_brokerage, 2)
)

print("\nGENERATED FILES")
print("---------------")
print("- top200.csv")
print("- model_comparison.csv")
print("- evaluation_summary.csv")
print("- feature_importance.csv")

print("\nPipeline executed successfully!")
