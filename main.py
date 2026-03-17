import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = pd.read_csv("dataset.csv")

# 2. Separate input and output
X = data.drop("disease", axis=1)
y = data["disease"]

# 3. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Check accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# 6. Take user input
print("\n--- Disease Prediction ---")
print("Enter 1 for YES, 0 for NO\n")

fever    = int(input("Do you have Fever?    "))
cough    = int(input("Do you have Cough?    "))
fatigue  = int(input("Do you have Fatigue?  "))
headache = int(input("Do you have Headache? "))
nausea   = int(input("Do you have Nausea?   "))

# 7. Predict
input_data = [[fever, cough, fatigue, headache, nausea]]
result = model.predict(input_data)

print("\n Predicted Disease:", result[0])