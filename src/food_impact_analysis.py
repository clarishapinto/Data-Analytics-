
# Food Impact on Indians - Machine Learning Analysis
# Author: Clarisha Lucia Pinto

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("../data/food_impact_india.csv")

# Handle missing values
for col in data.columns:
    if data[col].dtype == "object":
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].median(), inplace=True)

# Encode categorical variables
encoder = LabelEncoder()
categorical_cols = [
    "Gender", "Region", "Diet_Type", "Primary_Cuisine",
    "Spice_Level", "Health_Impact", "Exercise_Level"
]

for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col])

# Features and target
X = data.drop("Health_Score", axis=1)
y = data["Health_Score"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

# Visualization
plt.figure(figsize=(10,6))
sns.barplot(x=data["Daily_Calorie_Intake"], y=data["Health_Score"])
plt.title("Impact of Daily Calorie Intake on Health Score")
plt.xticks(rotation=90)
plt.show()
