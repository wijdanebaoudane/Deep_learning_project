from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO
import os

# Charger le modèle Keras
model = tf.keras.models.load_model('model_baoudane_fruit.h5')
target_size = model.input_shape[1:3]  # Modèle attend une entrée (height, width)

# Noms des classes (confirmer avec votre modèle)
class_names = ["apple", "banana", "cherry", "grapes", "kiwi", "mango", "orange", "pineapple", "strawberry", "watermelon"]

# Fonction pour prétraiter l'image
def preprocess_image(image: Image.Image, target_size=target_size):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.expand_dims(image, axis=0).astype('float32')
    return image

# Créer l'application FastAPI
app = FastAPI()

# Route d'accueil pour servir la page HTML
@app.get("/", response_class=HTMLResponse)
async def get_index():
    # Vérifiez si le fichier HTML existe et le lire en toute sécurité
    template_path = "template/index.html"
    if not os.path.exists(template_path):
        return JSONResponse(content={"error": "HTML template not found"}, status_code=404)
    
    with open(template_path, "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Route pour classifier une image
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Lire le fichier image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))
        
        # Prétraiter l'image
        processed_image = preprocess_image(image)
        
        # Faire la prédiction
        prediction = model.predict(processed_image)
        
        # Convertir la prédiction et la confiance
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100  # Convertir numpy.float32 en float
        
        # Retourner la réponse
        return JSONResponse(content={
            "prediction": predicted_class,
            "confidence": round(confidence, 2)
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
