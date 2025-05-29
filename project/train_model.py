import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load dataset
data = pd.read_csv('gesture_data.csv')

# Features and label
X = data.drop('label', axis=1).values
y = data['label'].values

# Indices for left and right hand features
right_hand_idx = slice(0, 63)
left_hand_idx = slice(63, 126)

# Function to check if hand landmarks are zero (no hand)
def is_hand_present(features, idx):
    return not np.all(features[idx] == 0)

# Split dataset into one-hand and two-hands
one_hand_rows = []
two_hands_rows = []

for i, features in enumerate(X):
    right_present = is_hand_present(features, right_hand_idx)
    left_present = is_hand_present(features, left_hand_idx)
    if right_present and left_present:
        two_hands_rows.append(i)
    elif right_present or left_present:
        one_hand_rows.append(i)
    # else: no hands detected? skip or treat separately

# Prepare datasets
X_one = X[one_hand_rows]
y_one = y[one_hand_rows]

X_two = X[two_hands_rows]
y_two = y[two_hands_rows]

# Encode labels
le_one = LabelEncoder()
y_one_enc = le_one.fit_transform(y_one)

le_two = LabelEncoder()
y_two_enc = le_two.fit_transform(y_two)

# Scale features
scaler_one = StandardScaler()
X_one_scaled = scaler_one.fit_transform(X_one)

scaler_two = StandardScaler()
X_two_scaled = scaler_two.fit_transform(X_two)

# Train models
model_one = RandomForestClassifier(n_estimators=100, random_state=42)
model_one.fit(X_one_scaled, y_one_enc)

model_two = RandomForestClassifier(n_estimators=100, random_state=42)
model_two.fit(X_two_scaled, y_two_enc)

# Evaluate one-hand model
X1_train, X1_test, y1_train, y1_test = train_test_split(X_one_scaled, y_one_enc, test_size=0.2, random_state=42)
model_one.fit(X1_train, y1_train)
y1_pred = model_one.predict(X1_test)
print("One-hand model classification report:")
print(classification_report(y1_test, y1_pred, target_names=le_one.classes_))

# Evaluate two-hands model
X2_train, X2_test, y2_train, y2_test = train_test_split(X_two_scaled, y_two_enc, test_size=0.2, random_state=42)
model_two.fit(X2_train, y2_train)
y2_pred = model_two.predict(X2_test)
print("Two-hands model classification report:")
print(classification_report(y2_test, y2_pred, target_names=le_two.classes_))

# Save models
joblib.dump(model_one, 'gesture_model_one_hand.pkl')
joblib.dump(scaler_one, 'scaler_one_hand.pkl')
joblib.dump(le_one, 'label_encoder_one_hand.pkl')

joblib.dump(model_two, 'gesture_model_two_hands.pkl')
joblib.dump(scaler_two, 'scaler_two_hands.pkl')
joblib.dump(le_two, 'label_encoder_two_hands.pkl')

print("✅ Both models trained and saved successfully!")
print("One-hand model accuracy:", model_one.score(X1_test, y1_test))
print("Two-hands model accuracy:", model_two.score(X2_test, y2_test))
