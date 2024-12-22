import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf

# Charger le modèle
MODEL_PATH = "model_LSTM_baoudane.h5"
model = load_model(MODEL_PATH)

# Titre de l'application
st.title("Application de Prédiction des Prix d'Actions")
st.markdown("Cette application prédit les prix des actions en utilisant un modèle LSTM.")

# Sélection du symbole de l'action et de la plage de dates
st.sidebar.header("Paramètres")
stock_symbol = st.sidebar.text_input("Symbole de l'action", value="AAPL", max_chars=10)
start_date = st.sidebar.date_input("Date de début", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("Date de fin", value=pd.to_datetime("2024-11-21"))

# Charger les données depuis Yahoo Finance
st.write(f"Chargement des données pour {stock_symbol}...")
try:
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    if data.empty:
        st.error("Aucune donnée disponible pour ce symbole ou cette plage de dates.")
    else:
        st.write("Données téléchargées :")
        st.dataframe(data.head())

        # Vérifier que la colonne 'Close' existe
        if 'Close' not in data.columns:
            st.error("Les données téléchargées ne contiennent pas la colonne 'Close'.")
        else:
            # Prétraitement des données
            st.write("Prétraitement des données...")
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_scaled = scaler.fit_transform(data[['Close']])

            # Création des séquences pour la prédiction
            time_step = 60
            X = []
            for i in range(time_step, len(data_scaled)):
                X.append(data_scaled[i - time_step:i, 0])
            X = np.array(X).reshape(-1, time_step, 1)

            # Faire des prédictions
            st.write("Prédictions en cours...")
            predictions = model.predict(X)
            predictions = scaler.inverse_transform(predictions)

            # Ajouter les prédictions au DataFrame d'origine
            data['Predicted'] = np.nan
            data.iloc[time_step:, data.columns.get_loc('Predicted')] = predictions.flatten()

            # Afficher les résultats
            st.write("Données avec Prédictions :")
            st.dataframe(data.tail())

            # Visualisation des résultats
            st.write("Visualisation des résultats :")
            plt.figure(figsize=(10, 6))
            plt.plot(data['Close'], label='Prix réel', color='blue')
            plt.plot(data['Predicted'], label='Prix prédit', color='orange')
            plt.title(f"Prédiction des Prix d'Actions pour {stock_symbol}")
            plt.xlabel("Temps")
            plt.ylabel("Prix d'Action")
            plt.legend()
            st.pyplot(plt)

except Exception as e:
    st.error(f"Erreur lors du téléchargement ou du traitement des données : {e}")
