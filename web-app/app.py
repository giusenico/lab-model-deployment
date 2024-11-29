import os
import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Global variables for model and encoder
model = None
encoder = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/train', methods=['POST'])
def train():
    global model, encoder
    
    # Load dataset
    ufos = pd.read_csv('web-app/data/ufos.csv')
    ufos = pd.DataFrame({
        'Seconds': ufos['duration (seconds)'],
        'Country': ufos['country'],
        'Latitude': ufos['latitude'],
        'Longitude': ufos['longitude']
    })
    ufos.dropna(inplace=True)
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    # Encode countries
    encoder = LabelEncoder()
    ufos['Country'] = encoder.fit_transform(ufos['Country'])
    
    # Prepare features and target
    selected_features = ['Seconds', 'Latitude', 'Longitude']
    X = ufos[selected_features]
    y = ufos['Country']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Save the model and encoder
    pickle.dump(model, open('web-app/ufo-model.pkl', 'wb'))
    pickle.dump(encoder, open('web-app/label_encoder.pkl', 'wb'))
    
    return {"message": "Model trained and saved successfully!"}, 200

@app.route("/predict", methods=["POST"])
def predict():
    global model, encoder
    
    # Check if the model is loaded
    if model is None or encoder is None:
        # Try to load the model and encoder
        try:
            model = pickle.load(open('web-app/ufo-model.pkl', 'rb'))
            encoder = pickle.load(open('web-app/label_encoder.pkl', 'rb'))
        except FileNotFoundError:
            return render_template("index.html", prediction_text="Model not trained yet. Please train the model first.")
    
    # Get features from form input
    try:
        int_features = [float(x) for x in request.form.values()]  # Handle float inputs
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
        output = encoder.inverse_transform(prediction)[0]  # Decode the prediction
        
        return render_template(
            "index.html", prediction_text=f"Likely country: {output}"
        )
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
