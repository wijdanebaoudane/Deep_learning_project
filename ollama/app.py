import streamlit as st
import requests
import json

# Endpoint correct pour Ollama
OLLAMA_API_URL = "http://localhost:11434/api/chat"

def main():
    st.title("Ollama avec Streamlit")
    user_input = st.text_area("Entrez votre prompt :", placeholder="Tapez votre texte ici...")

    if st.button("Envoyer à Ollama"):
        if user_input.strip() == "":
            st.warning("Veuillez entrer un texte avant d'envoyer.")
        else:
            with st.spinner("Communication avec Ollama..."):
                try:
                    # Requête POST à l'API Ollama
                    data = {
                        "model": "llama3.2:1b",
                        "messages": [{"role": "user", "content": user_input}]
                    }
                    response = requests.post(
                        OLLAMA_API_URL,
                        headers={"Content-Type": "application/json"},
                        data=json.dumps(data),
                        stream = True  # Pour gérer les réponses en flux
                    )

                    if response.status_code == 200:
                        full_response = ""
                        for line in response.iter_lines():
                            if line:
                                json_line = json.loads(line)
                                #On regarde si l'api nous a donné un chunk avec du contenu.
                                if 'message' in json_line and 'content' in json_line['message']:
                                    full_response += json_line['message']['content']

                        st.success("Réponse d'Ollama :")
                        st.text(full_response)  # Affiche toute la réponse
                        st.json(json.loads(line))#Affiche le dernier json (pour déboggage)
                    else:
                        st.error(f"Erreur {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Une erreur s'est produite : {e}")

if __name__ == "__main__":
    main()