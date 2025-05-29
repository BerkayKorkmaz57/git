import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load the dataset
data = pd.read_csv('gesture_data.csv')

# Verify correct number of columns (should be 127 = 1 label + 126 features)
expected_cols = 1 + 21 * 3 * 2  # label + 2 hands
if data.shape[1] != expected_cols:
    raise ValueError(f"Expected {expected_cols} columns (1 label + 126 features), but got {data.shape[1]}")

# Split features and label
X = data.drop('label', axis=1)
y = data['label']

# Encode labels (e.g., click → 0, drag → 1, etc.)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y_encoded)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))
# Save model components
joblib.dump(model, 'gesture_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("✅ Model trained and saved successfully!")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))