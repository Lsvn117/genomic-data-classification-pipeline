from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained SVM and PCA model
model = joblib.load('svm_model.pkl')
pca = joblib.load('pca_transformer.pkl')  # make sure this file exists

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'embedding_file' not in request.files:
        return "No file uploaded", 400

    file = request.files['embedding_file']
    if file.filename == '':
        return "No selected file", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        # Load embedding and reshape
        embedding = np.load(filepath)
        embedding_flat = embedding.reshape(1, -1)

        # Apply PCA
        embedding_pca = pca.transform(embedding_flat)

        # Predict using SVM
        prediction = model.predict(embedding_pca)[0]
        probability = model.predict_proba(embedding_pca)[0][prediction]

        label = "Pathogenic" if prediction == 1 else "Benign"
        result = f"Prediction: {label}"

    except Exception as e:
        return f"Error during prediction: {e}", 500

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
