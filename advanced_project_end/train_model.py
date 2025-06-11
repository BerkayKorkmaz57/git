# Save this as train_model.py
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler, LabelEncoder #To scale the data § point->0  
import joblib  # To save and load the trained model.

data = pd.read_csv('gesture_data.csv')
X = data.drop('label', axis=1)
y = data['label'] 

le = LabelEncoder() 
y = le.fit_transform(y)

scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier() 
model.fit(X_scaled, y)

joblib.dump(model, 'gesture_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Model trained and saved!")