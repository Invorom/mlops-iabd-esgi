"""Frontend Streamlit pour tester l'API de détection de fraude."""
from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Détection de Fraude", layout="wide", page_icon="💳")
st.title("💳 Outil de Détection de Fraude Bancaire par NEROT Romain")

home_tab, predict_tab = st.tabs(["Accueil", "Prédiction"])

with home_tab:
    st.header("Bienvenue sur le portail de détection")
    st.markdown("""
    Cette interface vous permet de tester en temps réel notre modèle d'intelligence artificielle conçu pour détecter les fraudes à la carte bancaire.
    
    ### 🎯 Objectif du projet
    Chaque année, les fraudes causent des milliards de pertes. L'enjeu est de bloquer les transactions frauduleuses avec précision, **sans** décliner les transactions légitimes (faux positifs) qui dégradent l'expérience client.
    
    ### 🚀 Comment utiliser cet outil ?
    1. Allez dans l'onglet **Prédiction**.
    2. Remplissez les informations de la transaction simulée (montant, pays, type de carte, etc.).
    3. Lancez l'analyse pour interroger l'API distante et obtenir le verdict (probabilité de fraude).
    
    *Ce projet est propulsé par FastAPI, MLflow, Optuna et Streamlit.*
    """)
    st.info("👈 Cliquez sur l'onglet **Prédiction** ci-dessus pour commencer.")

with predict_tab:
    st.subheader("Entrez les détails de la transaction")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Transaction")
            amount = st.number_input("Montant (Amount)", min_value=0.0, value=150.0, step=10.0)
            type_of_transaction = st.selectbox("Type de Transaction", ["POS", "Online", "ATM"])
            entry_mode = st.selectbox("Mode de saisie (Entry Mode)", ["Tap", "PIN", "CVC"])
            merchant_group = st.text_input("Catégorie marchand", value="Electronics")
            
        with col2:
            st.markdown("### Informations Client")
            age = st.number_input("Âge du titulaire", min_value=18, max_value=120, value=35, step=1)
            gender = st.selectbox("Genre", ["M", "F"])
            country_residence = st.text_input("Pays de résidence", value="United Kingdom")
            country_transaction = st.text_input("Pays de transaction", value="United Kingdom")
            
        with col3:
            st.markdown("### Carte et Banque")
            bank = st.text_input("Banque émettrice", value="HSBC")
            type_of_card = st.selectbox("Type de carte", ["Visa", "MasterCard"])
            day_of_week = st.selectbox("Jour de la transaction", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

        st.markdown("---")
        submitted = st.form_submit_button("🔍 Analyser la transaction", use_container_width=True)

    if submitted:
        payload = {
            "Amount": amount,
            "Age": age,
            "Day of Week": day_of_week,
            "Type of Card": type_of_card,
            "Entry Mode": entry_mode,
            "Type of Transaction": type_of_transaction,
            "Merchant Group": merchant_group,
            "Country of Transaction": country_transaction,
            "Country of Residence": country_residence,
            "Gender": gender,
            "Bank": bank
        }
        
        with st.spinner("Analyse en cours..."):
            try:
                response = httpx.post(f"{API_URL}/predict", json=payload, timeout=10.0)
                response.raise_for_status()
                result = response.json()
                
                prediction = result["prediction"]
                probability = result["probability"]
                
                if prediction == 1:
                    st.error(f"⚠️ Alerte ! Transaction potentiellement frauduleuse (Probabilité de fraude : {probability:.2%})")
                else:
                    st.success(f"✅ Transaction légitime (Probabilité de fraude : {probability:.2%})")
                
            except httpx.HTTPError as exc:
                st.error(f"Appel à l'API impossible : {exc}")
            except KeyError:
                st.error(f"Réponse inattendue de l'API : {result}")
