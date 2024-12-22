import os
import numpy as np
from django.shortcuts import render
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from .forms import ImageUploadForm

# Charger le modèle
MODEL_PATH = os.path.join(settings.BASE_DIR, 'classify', 'model_baoudane_fruit.h5')
model = load_model(MODEL_PATH)

# Définir les étiquettes
LABELS =  ['apple', 'banana', 'cherry', 'grapes', 'kiwi', 'mango', 'orange', 'pineapple', 'strawberry', 'watermelon']

def classify_fruit(request):
    prediction = None
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Sauvegarder l'image téléchargée
            image = form.cleaned_data['image']
            image_path = os.path.join(settings.MEDIA_ROOT, 'uploads', image.name)
            with open(image_path, 'wb') as f:
                for chunk in image.chunks():
                    f.write(chunk)

            # Prétraitement de l'image : redimensionner à 32x32
            img = load_img(image_path, target_size=(32, 32))  # Redimensionner à 32x32 pixels
            img_array = img_to_array(img)  # Convertir l'image en tableau numpy
            img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension pour le batch

            # Prédiction
            predictions = model.predict(img_array)  # Le modèle fait la prédiction
            prediction = LABELS[np.argmax(predictions)]  # Récupérer l'étiquette avec la probabilité la plus élevée

            # Supprimer l'image après traitement
            os.remove(image_path)
    else:
        form = ImageUploadForm()

    return render(request, 'classify/classify.html', {'form': form, 'prediction': prediction})
