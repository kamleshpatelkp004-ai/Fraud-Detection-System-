import pandas as pd
import numpy as np
import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("fraud_data.csv")
print(df.head())

le = LabelEncoder()
df["TransactionType"] = le.fit_transform(df["TransactionType"])

X = df.drop("IsFraud", axis=1)
y = df["IsFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

new_transaction = np.array([[60000, 1, 2, 1]]) 

# Amount, TransactionType, AccountAge, IsInternational

prediction = model.predict(new_transaction)

if prediction[0] == 1:
    print("⚠️ Fraudulent Transaction")
else:
    print("✅ Legitimate Transaction")