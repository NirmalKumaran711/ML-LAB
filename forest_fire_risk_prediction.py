# Forest Fire Risk Prediction Using Random Forest
# Keep forest_fire_dataset.csv in the same folder.

# ---------- Import the Required Libraries ----------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

print("Libraries imported successfully!")

# ---------- Load Dataset ----------
DATA_PATH = "forest_fire_dataset.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "forest_fire_dataset.csv was not found. "
        "Keep the CSV in the same folder as this notebook."
    )

df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print(df.head().to_string(index=False))

# ---------- Check Dataset Information ----------
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# ---------- Statistical Summary ----------
print(df.describe().round(2).T[["mean", "min", "max"]])

# ---------- Visualize Fire / No Fire Distribution ----------
counts = df["fire"].value_counts().sort_index()
print(counts.rename({0: "No Fire", 1: "Fire"}))

plt.figure(figsize=(7, 5))
plt.bar(["No Fire", "Fire"], counts.values, color=["#4c9f70", "#d9534f"], edgecolor="black")
plt.title("Distribution of Forest Fire Classes")
plt.xlabel("Class")
plt.ylabel("Count")
plt.grid(axis="y", alpha=0.2)
plt.show()

# ---------- Correlation Analysis ----------
corr = df.corr(numeric_only=True)

plt.figure(figsize=(9, 7))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ---------- Split Input and Output ----------
X = df.drop(columns=["fire"])
y = df["fire"]

print("Input features:")
for feature in X.columns:
    print("-", feature)

print("\nX shape:", X.shape)
print("y shape:", y.shape)

# ---------- Train-Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)

# ---------- Train Random Forest Classifier ----------
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("Random Forest model trained successfully!")

# ---------- Make Predictions ----------
y_pred = rf_model.predict(X_test)

comparison = pd.DataFrame({
    "Actual": np.where(y_test.values[:10] == 1, "Fire", "No Fire"),
    "Predicted": np.where(y_pred[:10] == 1, "Fire", "No Fire")
})
print(comparison.to_string(index=False))

# ---------- Model Evaluation ----------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"Accuracy in percentage: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Fire", "Fire"]))

# ---------- Confusion Matrix ----------
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks([0, 1], ["No Fire", "Fire"])
plt.yticks([0, 1], ["No Fire", "Fire"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ---------- Feature Importance ----------
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=True)

print(importance_df.sort_values("Importance", ascending=False).round(6).to_string(index=False))

plt.figure(figsize=(9, 5))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.show()

# ---------- Predict Forest Fire Risk for New Conditions ----------
def predict_fire_risk(temperature, humidity, wind_speed, rain,
                      ffmc, dmc, dc, isi, bui, fwi):
    area = pd.DataFrame([{
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "rain": rain,
        "ffmc": ffmc,
        "dmc": dmc,
        "dc": dc,
        "isi": isi,
        "bui": bui,
        "fwi": fwi
    }])

    probability = float(rf_model.predict_proba(area)[0][1]) * 100

    if probability >= 70:
        level = "High Risk"
    elif probability >= 40:
        level = "Moderate Risk"
    else:
        level = "Low Risk"

    return round(probability, 2), level


# Hot, dry and windy day with no rain
prob, level = predict_fire_risk(
    temperature=38, humidity=45, wind_speed=20, rain=0.0,
    ffmc=90, dmc=45, dc=180, isi=4.5, bui=45, fwi=3.5
)
print("Fire Probability (%):", prob)
print("Fire Risk Level:", level)

# Cool, humid day with rain
prob, level = predict_fire_risk(
    temperature=25, humidity=85, wind_speed=10, rain=6.0,
    ffmc=60, dmc=8, dc=30, isi=0.3, bui=9, fwi=0.1
)
print("\nFire Probability (%):", prob)
print("Fire Risk Level:", level)

# ---------- Save Trained Model ----------
joblib.dump(rf_model, "forest_fire_random_forest.joblib")
print("Model saved successfully!")

