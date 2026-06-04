# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

# Load dataset
df = pd.read_csv("processes.csv")

# Remove unnecessary column
df.drop("Unnamed: 0", axis=1, inplace=True)

# Features and Target
X = df.drop("selling_price", axis=1)
y = df["selling_price"]

# Categorical columns
categorical_cols = [
    "name",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "Mileage Unit"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

# Random Forest Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
pipeline.fit(X_train, y_train)

# Prediction
y_pred = pipeline.predict(X_test)

# Accuracy
score = r2_score(y_test, y_pred)
print("R2 Score:", score)

# Save model
joblib.dump(pipeline, "car_price_model.pkl")

print("Model Saved Successfully!")