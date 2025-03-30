from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Initialize Flask app
app = Flask( template_folder="templates")

# Extended dataset with more diseases and symptoms
data = {
    "fever": [1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
    "cough": [1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
    "headache": [0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
    "fatigue": [0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
    "chest_pain": [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
    "runny_nose": [1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    "sore_throat": [1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    "shortness_of_breath": [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
    "nausea": [0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
    "vomiting": [0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
    "diarrhea": [0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
    "dizziness": [0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
    "body_ache": [1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    "disease": ["Flu", "Cold", "Migraine", "Fatigue Syndrome", "Heart Problem", "COVID-19", "Asthma", "Pneumonia", "Food Poisoning", "Dehydration"]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Train model
X = df.drop(columns=["disease"])
y = df["disease"]
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
with open("HEALTH CHAT BOT/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load trained model
with open("HEALTH CHAT BOT/model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the list of symptoms used in training
symptoms_list = list(X.columns)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    symptoms = request.json.get("symptoms", [])  # Get symptoms from user input
    input_data = {symptom: 1 if symptom in symptoms else 0 for symptom in symptoms_list}  
    X_input = pd.DataFrame([input_data])
    
    prediction = model.predict(X_input)[0]  # Predict disease
    return jsonify({"disease": prediction})

if healthchatbot == "main":
    app.run(debug=True)