import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("processes.csv")

df = df.drop("Unnamed: 0", axis=1)

X = df[[
    "year",
    "km_driven",
    "seats",
    "Mileage",
    "Engine (CC)",
    "max_power (in bph)"
]]

y = df["selling_price"]

model = RandomForestRegressor()
model.fit(X, y)

importance = model.feature_importances_

plt.figure(figsize=(8,5))
plt.bar(X.columns, importance)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.show()