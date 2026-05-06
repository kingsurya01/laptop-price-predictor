import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("laptop_data.csv")

print(df.head())

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

df["Ram"] = df["Ram"].str.replace("GB", "").astype(int)
df["Weight"] = df["Weight"].str.replace("kg", "").astype(float)

df["Price_per_RAM"] = df["Price"] / df["Ram"]

price_array = np.array(df["Price"])

print("Mean Price:", np.mean(price_array))
print("Median Price:", np.median(price_array))
print("Standard Deviation:", np.std(price_array))

correlation_matrix = df.corr(numeric_only=True)
print(correlation_matrix["Price"].sort_values(ascending=False))

if "Laptop_Name" in df.columns:
    df = df.drop("Laptop_Name", axis=1)

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



plt.figure()
plt.scatter(df["Ram"], df["Price"])
plt.title("RAM vs Price")
plt.xlabel("RAM (GB)")
plt.ylabel("Price")
plt.show()



model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

plt.figure()
n = 20

plt.plot(range(n), y_test.values[:n], label="Actual")
plt.plot(range(n), y_pred[:n], label="Predicted")

plt.xlabel("Sample Index")
plt.ylabel("Price")
plt.title("Actual vs Predicted Prices")
plt.legend()
plt.show()
