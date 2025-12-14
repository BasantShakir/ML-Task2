
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv("calories.csv")
df.head()

numeric_features = ["Age", "Weight", "Height", "Duration", "Heart_Rate", "Body_Temp"]
categorical_features = ["Activity"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(), categorical_features)
])
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])
X = df.drop("Calories", axis=1)
y = df["Calories"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
sample = pd.DataFrame([{
    "Age": 25,
    "Weight": 70,
    "Height": 175,
    "Duration": 60,
    "Heart_Rate": 130,
    "Body_Temp": 98.6,
    "Activity": "Brisk Walking"
}])

predicted_calories = model.predict(sample)
print(f"Predicted Calories Burned: {predicted_calories[0]:.2f}")