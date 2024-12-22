import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Charger le modèle Keras
model = tf.keras.models.load_model('model_baoudane_fruit.h5')
target_size = model.input_shape[1:3]  # Modèle attend une entrée (height, width)

# Noms des classes (confirmer avec votre modèle)
class_names = ["apple", "banana", "cherry", "grapes", "kiwi", "mango", "orange", "pineapple", "strawberry", "watermelon"]

# Fonction pour prétraiter l'image
def preprocess_image(image, target_size=target_size):
    # Convertir en RGB (3 canaux)
    image = image.convert("RGB")
    # Redimensionner l'image
    image = image.resize(target_size)
    # Ajouter la dimension batch
    image = np.expand_dims(image, axis=0).astype('float32')
    # Normaliser les pixels
    return image

# Interface utilisateur Streamlit
st.set_page_config(page_title="Fruit Classifier", layout="wide")

# Définir un thème de couleur
st.markdown("""
    <style>
        .stApp {
            background-color: #f7f7f7;
        }
        .sidebar .sidebar-content {
            background-color: #ffeb3b;
            color: #333;
        }
        h1, h2, h3 {
            color: #ff5722;
        }
        .stButton>button {
            background-color: #ff5722;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .centered-image {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 20px;
        }
        .sidebar .profile-image {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar avec avatar à gauche
st.sidebar.markdown('<div class="profile-image">', unsafe_allow_html=True)
st.sidebar.image("image/image.png", width=100)  # Remplacer par votre image locale
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.write("### Welcome")
st.sidebar.write("Fruit Classifier - Classify your fruit images!")

# Menu déroulant avec avatar
menu_option = st.sidebar.selectbox(
    "Select Option",
    ("Home", "Profile", "Settings")  # Options disponibles
)

# Affichage du menu en fonction de l'option sélectionnée
if menu_option == "Home":
    st.title("Fruit Classifier")
    st.write("Upload an image of a fruit, and the model will predict its type!")
    
    # Téléchargement d'image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tiff", "gif"])

    if uploaded_file is not None:
        try:
            # Afficher l'image téléchargée dans un container centré
            with st.container():
                st.markdown('<div class="centered-image">', unsafe_allow_html=True)
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=False)
                st.markdown('</div>', unsafe_allow_html=True)

            st.write("Classifying...")

            # Prétraiter l'image et faire une prédiction
            image = Image.open(uploaded_file)
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)

            # Afficher la prédiction
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            # Présenter la prédiction avec des éléments de design
            st.markdown(f"### **Prediction**: {predicted_class}")
            st.markdown(f"**Confidence**: {confidence:.2f}%")
            st.progress(confidence / 100)

        except Exception as e:
            st.error(f"An error occurred: {e}")

elif menu_option == "Profile":
    st.title("User Profile")
    st.write("This is your profile section.")
    st.write("### Name: John Doe")
    st.write("### Email: john.doe@example.com")

elif menu_option == "Settings":
    st.title("Settings")
    st.write("This is the settings section.")
    st.write("You can add your settings here.")
