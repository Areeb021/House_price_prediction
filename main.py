# House Price Prediction - comparing 3 models (Linear Regression, Decision Tree, Random Forest)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------------
df = pd.read_csv("resources/data.csv")
print("Shape:", df.shape)
print(df.head())

# ---------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------
# remove columns that are not useful for prediction
df = df.drop(columns=["date", "street", "statezip", "country"])

# some houses have price = 0 (wrong data), remove them
df = df[df["price"] > 0]

# remove extreme price outliers (IQR method)
q1 = df["price"].quantile(0.25)
q3 = df["price"].quantile(0.75)
iqr = q3 - q1
df = df[(df["price"] >= q1 - 3 * iqr) & (df["price"] <= q3 + 3 * iqr)]

# convert the text column (city) into numbers
df = pd.get_dummies(df, columns=["city"], drop_first=True)

print("Shape after cleaning:", df.shape)

# ---------------------------------------------------------------
# 3. Look at the data (graph 1: price distribution)
# ---------------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.hist(df["price"], bins=40, color="steelblue")
plt.title("House Price Distribution")
plt.xlabel("Price")
plt.ylabel("Number of houses")
plt.tight_layout()
plt.savefig("1_price_distribution.png")
plt.show()

# ---------------------------------------------------------------
# 4. Split data into train (80%) and test (20%)
# ---------------------------------------------------------------
X = df.drop(columns=["price"])   # features
y = df["price"]                  # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------------
# 5. Train the 3 models
# ---------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

results = []        # to store the scores
predictions = {}    # to store predictions for the graphs

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()

    results.append([name, mae, rmse, r2, cv])

# ---------------------------------------------------------------
# 6. Compare the models
# ---------------------------------------------------------------
results = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2", "CV R2"])
print("\nModel Comparison:")
print(results.round(3).to_string(index=False))
results.to_csv("model_comparison.csv", index=False)

colors = ["tab:blue", "tab:orange", "tab:green"]

# graph 2: bar charts of the scores
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
    ax.bar(results["Model"], results[metric], color=colors)
    ax.set_title(metric + (" (higher is better)" if metric == "R2" else " (lower is better)"))
    ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig("2_model_comparison.png")
plt.show()

# graph 3: actual vs predicted price for each model
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, pred), c in zip(axes, predictions.items(), colors):
    ax.scatter(y_test, pred, alpha=0.4, s=10, color=c)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax.set_title(name)
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
plt.tight_layout()
plt.savefig("3_actual_vs_predicted.png")
plt.show()

# ---------------------------------------------------------------
# 7. Feature importance of Random Forest (graph 4)
# ---------------------------------------------------------------
rf = models["Random Forest"]
importance = pd.Series(rf.feature_importances_, index=X.columns)
top10 = importance.sort_values(ascending=False).head(10)
print("\nTop 10 important features:")
print(top10.round(3))

plt.figure(figsize=(8, 5))
top10[::-1].plot(kind="barh", color="green")
plt.title("Random Forest - Top 10 Important Features")
plt.tight_layout()
plt.savefig("4_feature_importance.png")
plt.show() 