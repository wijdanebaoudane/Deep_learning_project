from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Initialisation de Flask
app = Flask(__name__)

# Charger le modèle (Assurez-vous que le chemin est correct)
model_path = r'C:\Users\You\Desktop\Baoudane_DeepLearning\baoudane_venv\Lab2\flask\model_baoudane_fruit.h5'

# Vérifiez si le fichier existe avant de charger
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Le fichier du modèle est introuvable à l'emplacement {model_path}")

# Charger le modèle
try:
    model = tf.keras.models.load_model(model_path)
    print("Modèle chargé avec succès")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {e}")

# Classe associée à vos prédictions (à adapter selon votre modèle)
classes = ['apple', 'banana', 'cherry', 'grapes', 'kiwi', 'mango', 'orange', 'pineapple', 'strawberry', 'watermelon']  # Exemple de classes

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier n'a été téléchargé."})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Aucun fichier sélectionné."})

    try:
        # Lire l'image
        image = Image.open(file).convert('RGB')

        # Redimensionner l'image à 32x32 pixels (changer la taille ici)
        image = image.resize((32, 32))  # Redimensionner l'image à 32x32

        # Convertir l'image en tableau numpy sans normalisation
        image_array = np.array(image)  # Pas de normalisation ici
        image_array = np.expand_dims(image_array, axis=0)  # Ajouter la dimension batch

        # Faire la prédiction
        predictions = model.predict(image_array)

        # Trouver la classe prédite avec la plus haute probabilité
        predicted_class = classes[np.argmax(predictions)]
        confidence = float(np.max(predictions)) * 100  # Conversion en pourcentage

        # Afficher la page avec les résultats
        return render_template('index.html', prediction=predicted_class, confidence=confidence)

    except Exception as e:
        return jsonify({"error": f"Une erreur est survenue lors du traitement de l'image : {str(e)}"})

# Lancer l'application
if __name__ == '__main__':
    app.run(debug=True)
