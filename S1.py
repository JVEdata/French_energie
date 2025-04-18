# -*- coding: utf-8 -*- # Pour assurer la compatibilité des caractères spéciaux
import streamlit as st
import pandas as pd
from PIL import Image
import io
import base64
import numpy as np
import os
import time # Optional: To demonstrate loading time difference
import traceback # For detailed error reporting
import joblib # <-- AJOUT : Pour charger les modèles .joblib
import json   # <-- AJOUT : Pour charger les fichiers .json
import datetime # <-- AJOUT : Pour manipuler les dates
import gdown # <-- AJOUTÉ : Pour télécharger depuis Google Drive

# =============================================================================
# --- 1. CONFIGURATION ---
# =============================================================================

# --- LOCAL FILE PATHS ---
# Les chemins GDrive restent inchangés
ECO2MIX_CSV_PATH = 'https://drive.google.com/file/d/1hvEXS7ABSGUh45QHvoR1aCKzJzsQ0S4B/view?usp=drive_link'
SOUTIRAGE_CSV_PATH = 'https://drive.google.com/file/d/1vIsX-TemlEYBTH9dNA4vaEY_6lcYiX6j/view?usp=drive_link'

# <-- MODIFIÉ : Chemins relatifs depuis la racine du projet (streamlit_app.py) -->
EFFECTIFS_CSV_PATH = '2. base_etablissement_par_tranche_effectif.csv'
TEMPERATURE_CSV_PATH = '3. temperature-quotidienne-regionale.csv'
POPULATION_CSV_PATH = '4. Population - Insee.csv'
DF_FINAL_CSV_PATH = '6. df_final.csv'
CORRELATION_MATRIX_IMAGE_PATH = 'Visualisation/matrice.png'
MODEL_PREDICTION_IMAGE_PATH = 'Visualisation/modele.png'

# --- AJOUT: CHEMINS DES ARTEFACTS ML ---
# <-- MODIFIÉ : Chemin relatif pour le dossier -->
ML_ARTIFACTS_DIR = 'Machine_learning' # Dossier relatif à streamlit_app.py
BEST_PIPELINE_PATH = os.path.join(ML_ARTIFACTS_DIR, 'best_pipeline_rf.joblib')
COLUMNS_INFO_PATH = os.path.join(ML_ARTIFACTS_DIR, 'columns_info.json')
REGIONS_PATH = os.path.join(ML_ARTIFACTS_DIR, 'regions.json')
# Note: Pas besoin de charger model_rf.joblib et preprocessor.joblib séparément
# car best_pipeline_rf.joblib contient déjà le pipeline complet (préprocesseur + modèle)

# --- AJOUT CHEMIN LOGO ---
# <-- MODIFIÉ : Chemin relatif -->
LOGO_PATH = 'Visualisation/logo.png' # Chemin relatif à streamlit_app.py

# --- Section Icons ---
SECTION_ICONS = {
    "👋 Introduction": "👋 Introduction",
    "🔎 Exploration des données": "🔎 Exploration des données",
    "📊 Data Visualisation": "📊 Data Visualisation",
    # "🛠️ Preprocessing des Données 🛠️": "🛠️ Preprocessing", # Commenté ou supprimé si plus dans le menu principal
    "⚙️ Modélisation": "⚙️ Modélisation",
    "🤖 Prédiction": "🤖 Prédiction", # <-- AJOUTÉ DANS LE DICTIONNAIRE
    "📌 Conclusion": "📌 Conclusion"
}

# =============================================================================
# --- 2. STREAMLIT APP CONFIGURATION ---
# =============================================================================

st.set_page_config(
    page_title="Consommation d'Électricité en France",
    page_icon="⚡",
    layout="wide"
)
# =============================================================================
# --- 3. SIDEBAR ---
# =============================================================================

import base64
import os
import streamlit as st

with st.sidebar:
    # --- CSS POUR RÉDUIRE LA LARGEUR DE LA SIDEBAR ---
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                width: 305px !important;
                min-width: 305px !important;
                max-width: 305px !important;
            }

            .css-18e3th9 {
                margin-left: 305px !important;
            }

            [data-testid="stSidebar"] > div {
                padding: 0.5rem;
            }

            .custom-signature {
                font-size: 0.9rem;
                text-align: center;
                margin-top: 1.5rem;
                line-height: 1.4;
            }

            .about-label {
                font-size: 0.75rem;
                color: #888888;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 0.3rem;
            }
        </style>
    """, unsafe_allow_html=True)

    # --- TITRE ET MENU ---
    st.markdown("<h1 style='color: #5533FF;'>📚 Sommaire</h1>", unsafe_allow_html=True)
    st.markdown("Aller vers 👇")

    sidebar_options = list(SECTION_ICONS.keys())
    if "🛠️ Preprocessing des Données 🛠️" in sidebar_options:
        sidebar_options.remove("🛠️ Preprocessing des Données 🛠️")

    if 'choix' not in st.session_state:
        st.session_state.choix = sidebar_options[0]

    def update_choice():
        st.session_state.choix = st.session_state.choix_radio

    current_index = 0
    if st.session_state.choix in sidebar_options:
        current_index = sidebar_options.index(st.session_state.choix)
    else:
        st.session_state.choix = sidebar_options[0]

    choix = st.radio(
        "",
        sidebar_options,
        key='choix_radio',
        index=current_index,
        on_change=update_choice,
        format_func=lambda x: SECTION_ICONS.get(x, x)
    )

    # --- SÉPARATEUR ---
    st.markdown("<hr>", unsafe_allow_html=True)

    # --- LOGO ---
    if os.path.exists(LOGO_PATH):
        try:
            with open(LOGO_PATH, "rb") as image_file:
                encoded_logo = base64.b64encode(image_file.read()).decode()

            st.markdown(f"""
                <div style='margin-top: 2rem; text-align: center;'>
                    <img src='data:image/png;base64,{encoded_logo}' style='width:100%; max-width: 200px; margin-bottom: 1rem;'/>
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Erreur chargement logo: {e}")
    else:
        st.warning(f"Logo non trouvé : {LOGO_PATH}")
        print(f"Warning: Logo file not found at {LOGO_PATH}")

    # --- LIEN VERS LINKEDIN ---
    st.markdown("""
        <div style='text-align: center; margin-top: 1rem;'>
            <a href="https://www.linkedin.com/in/jeremy-vanerpe/" target="_blank" style="text-decoration: none; font-weight: bold; color: #0077B5;">
                👉 Contactez-moi sur LinkedIn
            </a>
        </div>
    """, unsafe_allow_html=True)

    # --- SIGNATURE / FONCTION ---
    st.markdown("""
        <div class="custom-signature" style="color: #CCCCCC;">
            <div class="about-label">À propos</div>
            <strong style="color: #DDDDDD;">Jérémy VAN ERPE</strong><br>
            Optimisation financière,<br>
            Analyse de données & Conseil
        </div>
    """, unsafe_allow_html=True)



# =============================================================================
# --- NOUVELLE SECTION: FONCTIONS DE CHARGEMENT DES ARTEFACTS ML ---
# =============================================================================


@st.cache_resource # Cache les objets non sérialisables comme les modèles/pipelines
def load_pipeline(path):
    """Charge le pipeline ML depuis un fichier .joblib."""
    print(f"--- Chargement du pipeline depuis {path} ---")
    try:
        pipeline = joblib.load(path)
        print("--- Pipeline chargé avec succès. ---")
        return pipeline
    except FileNotFoundError:
        st.error(f"ERREUR : Fichier pipeline introuvable : {path}")
        print(f"Error: Pipeline file not found: {path}")
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du pipeline ({path}): {e}")
        print(f"Error loading pipeline ({path}): {e}")
        traceback.print_exc()
        return None

@st.cache_data # Cache les données sérialisables comme les listes/dictionnaires JSON
def load_json_data(path):
    """Charge les données depuis un fichier JSON."""
    print(f"--- Chargement des données JSON depuis {path} ---")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"--- Données JSON chargées depuis {path}. ---")
        return data
    except FileNotFoundError:
        st.error(f"ERREUR : Fichier JSON introuvable : {path}")
        print(f"Error: JSON file not found: {path}")
        return None
    except json.JSONDecodeError:
        st.error(f"ERREUR : Le fichier {path} n'est pas un JSON valide.")
        print(f"Error: File {path} is not valid JSON.")
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du JSON ({path}): {e}")
        print(f"Error loading JSON ({path}): {e}")
        traceback.print_exc()
        return None

# --- Chargement effectif des artefacts (appel des fonctions cachées) ---
# Sera exécuté une seule fois grâce au cache
pipeline = load_pipeline(BEST_PIPELINE_PATH)
columns_info = load_json_data(COLUMNS_INFO_PATH)
regions_list = load_json_data(REGIONS_PATH)


# =============================================================================
# --- 4. INTERNAL DATA LOADING FUNCTIONS (with @st.cache_data) ---
# =============================================================================
# Fonctions internes préfixées par _ pour indiquer leur usage privé
# Elles contiennent la logique de chargement et de pré-traitement de base.

# /!\ MODIFICATION CI-DESSOUS POUR GOOGLE DRIVE /!\
@st.cache_data
def _load_eco2mix_internal(file_identifier):
    """Charge et prépare les données Eco2mix. Gère les URL GDrive ou les chemins locaux. Mis en cache."""
    print(f"--- Executing _load_eco2mix_internal for identifier: {file_identifier} ---")
    local_path_to_read = None
    temp_file_downloaded = False

    # Détecter si c'est une URL Google Drive
    if isinstance(file_identifier, str) and file_identifier.startswith('https://drive.google.com/'):
        # Définir un chemin de destination temporaire pour le téléchargement
        temp_eco2mix_path = '/tmp/eco2mix_temp_download.csv' # Utilise /tmp qui est généralement disponible
        print(f"--- Identifier is a Google Drive URL. Attempting download to {temp_eco2mix_path}... ---")
        try:
            # Supprimer l'ancien fichier temporaire s'il existe
            if os.path.exists(temp_eco2mix_path):
                os.remove(temp_eco2mix_path)
                print(f"--- Removed existing temp file: {temp_eco2mix_path} ---")

            # Télécharger le fichier depuis Google Drive
            gdown.download(url=file_identifier, output=temp_eco2mix_path, quiet=False, fuzzy=True) # fuzzy=True aide avec les gros fichiers
            print(f"--- Google Drive file downloaded successfully to {temp_eco2mix_path} ---")
            local_path_to_read = temp_eco2mix_path
            temp_file_downloaded = True # Marquer qu'un fichier temporaire a été créé

        except Exception as e_gdown:
            st.error(f"Erreur lors du téléchargement depuis Google Drive ({file_identifier}): {e_gdown}")
            print(f"Error: Failed to download from Google Drive URL {file_identifier}: {e_gdown}")
            traceback.print_exc()
            return None # Échec du chargement
    else:
        # Si ce n'est pas une URL GDrive, on suppose que c'est un chemin local
        print(f"--- Identifier is treated as a local path: {file_identifier} ---")
        if not os.path.exists(file_identifier):
             print(f"Error: Local file not found: {file_identifier}")
             st.error(f"Fichier local non trouvé: {file_identifier}") # Remettre l'erreur si chemin local
             return None
        local_path_to_read = file_identifier # Utiliser le chemin local directement

    # Vérifier si un chemin de lecture a été défini
    if local_path_to_read is None:
        print("Error: No valid local path determined for reading.")
        st.error("Impossible de déterminer le fichier à lire (problème de téléchargement ou chemin local invalide).")
        return None

    # --- Logique de lecture Pandas (inchangée, mais utilise local_path_to_read) ---
    df = None
    try:
        # Essai avec ; et utf-8
        df = pd.read_csv(local_path_to_read, sep=';', encoding='utf-8')
        print(f"--- Eco2mix loaded from '{local_path_to_read}' with sep=';', encoding='utf-8'.")
    except (UnicodeDecodeError, pd.errors.ParserError):
        print(f"Warning: Failed reading Eco2mix from '{local_path_to_read}' with semicolon/utf-8. Trying semicolon/latin-1...")
        try:
            df = pd.read_csv(local_path_to_read, sep=';', encoding='latin-1')
            print(f"--- Eco2mix loaded from '{local_path_to_read}' with sep=';', encoding='latin-1'.")
        except (UnicodeDecodeError, pd.errors.ParserError):
            print(f"Warning: Failed reading Eco2mix from '{local_path_to_read}' with semicolon/latin-1. Trying comma/utf-8...")
            try:
                df = pd.read_csv(local_path_to_read, sep=',', encoding='utf-8')
                print(f"--- Eco2mix loaded from '{local_path_to_read}' with sep=',', encoding='utf-8'.")
            except (UnicodeDecodeError, pd.errors.ParserError):
                print(f"Warning: Failed reading Eco2mix from '{local_path_to_read}' with comma/utf-8. Trying comma/latin-1...")
                try:
                    df = pd.read_csv(local_path_to_read, sep=',', encoding='latin-1')
                    print(f"--- Eco2mix loaded from '{local_path_to_read}' with sep=',', encoding='latin-1'.")
                except Exception as final_e:
                    print(f"Error: Final read error for Eco2mix from '{local_path_to_read}': {final_e}")
                    st.error(f"Erreur finale de lecture Eco2mix {local_path_to_read}: {final_e}")
                    traceback.print_exc()
                    return None
            except Exception as e_comma:
                 print(f"Error: Read error for Eco2mix from '{local_path_to_read}' with sep=',': {e_comma}")
                 st.error(f"Erreur de lecture Eco2mix {local_path_to_read} avec sep=',': {e_comma}")
                 traceback.print_exc()
                 return None
        except Exception as e_semicolon:
            print(f"Error: Read error for Eco2mix from '{local_path_to_read}' with sep=';': {e_semicolon}")
            st.error(f"Erreur de lecture Eco2mix {local_path_to_read} avec sep=';': {e_semicolon}")
            traceback.print_exc()
            return None
    except pd.errors.EmptyDataError:
         print(f"Warning: Eco2mix file {local_path_to_read} is empty.")
         st.warning(f"Le fichier Eco2mix {local_path_to_read} est vide.")
         df = pd.DataFrame() # Retourner un DF vide
    except FileNotFoundError: # Au cas où le fichier temp disparaîtrait entre le téléchargement et la lecture
        print(f"ERROR: File {local_path_to_read} not found unexpectedly.")
        st.error(f"ERREUR: Fichier non trouvé de manière inattendue à {local_path_to_read}")
        return None
    except Exception as e:
         print(f"Error: Unexpected error loading Eco2mix from '{local_path_to_read}': {e}")
         st.error(f"Erreur inattendue lors du chargement Eco2mix ({local_path_to_read}): {e}")
         traceback.print_exc()
         return None

    # Nettoyage du fichier temporaire SI il a été téléchargé
    # if temp_file_downloaded and os.path.exists(local_path_to_read):
    #     try:
    #         os.remove(local_path_to_read)
    #         print(f"--- Cleaned up temporary file: {local_path_to_read} ---")
    #     except Exception as e_clean:
    #         print(f"Warning: Could not clean up temporary file {local_path_to_read}: {e_clean}")

    if df is None and not isinstance(df, pd.DataFrame): # Si aucune méthode n'a fonctionné mais pas d'exception fatale ET pas de DF vide retourné
        print(f"Error: Cannot read Eco2mix file {local_path_to_read} with tested separators/encodings.")
        st.error(f"Impossible de lire le fichier Eco2mix {local_path_to_read} avec les séparateurs/encodages testés.")
        return None

    # Prétraitement de base (inchangé)
    if not df.empty:
        df = df.replace('ND', np.nan)
        if 'Date' in df.columns:
             df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if 'Consommation (MW)' in df.columns:
             df['Consommation (MW)'] = pd.to_numeric(df['Consommation (MW)'], errors='coerce')

    print(f"--- Eco2mix preprocessed. Shape: {df.shape} ---")
    return df
# /!\ FIN MODIFICATION ECO2MIX /!\


@st.cache_data
def _load_effectifs_internal(file_path):
    """Charge et prépare les données Effectifs. Mis en cache."""
    # st.toast(f"⏳ Chargement Effectifs depuis {file_path}...") # Moins de toasts
    print(f"--- Executing _load_effectifs_internal for {file_path} ---")
    if not os.path.exists(file_path):
        # st.error(f"Fichier Effectifs non trouvé: {file_path}") # Moins d'erreurs directes
        print(f"Error: Effectifs file not found: {file_path}")
        return None
    df = None
    try:
        # Essayer le séparateur virgule en premier
        df = pd.read_csv(file_path, sep=',', encoding='utf-8')
        print("--- Effectifs loaded successfully with sep=',' and encoding='utf-8'.")
    except (UnicodeDecodeError, pd.errors.ParserError):
        print(f"Warning: Failed reading Effectifs with comma/utf-8. Trying comma/latin-1...")
        try:
            df = pd.read_csv(file_path, sep=',', encoding='latin-1')
            print("--- Effectifs loaded successfully with sep=',' and encoding='latin-1'.")
        except (UnicodeDecodeError, pd.errors.ParserError):
            print(f"Warning: Failed reading Effectifs with comma/latin-1. Trying semicolon/utf-8...")
            # st.warning(f"Échec lecture Effectifs avec sep=','. Essai avec sep=';'...") # Moins de warnings directs
            try:
                df = pd.read_csv(file_path, sep=';', encoding='utf-8')
                print("--- Effectifs loaded successfully with sep=';' and encoding='utf-8'.")
            except (UnicodeDecodeError, pd.errors.ParserError):
                print(f"Warning: Failed reading Effectifs with semicolon/utf-8. Trying semicolon/latin-1...")
                try:
                    df = pd.read_csv(file_path, sep=';', encoding='latin-1')
                    print("--- Effectifs loaded successfully with sep=';' and encoding='latin-1'.")
                except Exception as final_e:
                    # st.error(f"Erreur finale de lecture Effectifs {file_path}: {final_e}")
                    print(f"Error: Final read error for Effectifs {file_path}: {final_e}")
                    traceback.print_exc()
                    return None
            except Exception as e_semi:
                 # st.error(f"Erreur de lecture Effectifs {file_path} avec sep=';': {e_semi}")
                 print(f"Error: Read error for Effectifs {file_path} with sep=';': {e_semi}")
                 traceback.print_exc()
                 return None
        except Exception as e_comma:
            # st.error(f"Erreur de lecture Effectifs {file_path} avec sep=',': {e_comma}")
            print(f"Error: Read error for Effectifs {file_path} with sep=',': {e_comma}")
            traceback.print_exc()
            return None
    except pd.errors.EmptyDataError:
        # st.warning(f"Le fichier Effectifs {file_path} est vide.")
        print(f"Warning: Effectifs file {file_path} is empty.")
        return pd.DataFrame()
    except FileNotFoundError:
        # st.error(f"ERREUR: Fichier Effectifs non trouvé à {file_path}")
        print(f"ERROR: Effectifs file not found at {file_path}")
        return None
    except Exception as e:
        # st.error(f"Erreur inattendue lors du chargement Effectifs ({file_path}): {e}")
        print(f"Error: Unexpected error loading Effectifs ({file_path}): {e}")
        traceback.print_exc()
        return None

    if df is None:
         # st.error(f"Impossible de lire le fichier Effectifs {file_path} avec les séparateurs/encodages testés.")
         print(f"Error: Cannot read Effectifs file {file_path} with tested separators/encodings.")
         return None

    if df.empty:
         # st.warning(f"Le fichier Effectifs {file_path} est vide ou n'a pas pu être chargé correctement.")
         print(f"Warning: Effectifs file {file_path} is empty or could not be loaded correctly.")
         return df # Retourne le DF vide

    # Prétraitement : Conversion des colonnes numériques
    num_cols = ['E14TST', 'E14TS0ND', 'E14TS1', 'E14TS6', 'E14TS10',
                'E14TS20', 'E14TS50', 'E14TS100', 'E14TS200', 'E14TS500']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Remplacer les NaN par 0 APRES conversion si c'est le comportement souhaité
            # df[col] = df[col].fillna(0)
            print(f"--- Effectifs: Column '{col}' converted to numeric.")
        else:
            # st.warning(f"Colonne numérique attendue '{col}' manquante dans Effectifs.")
            print(f"Warning: Expected numeric column '{col}' missing in Effectifs.")
            # Optionnel: Créer la colonne avec 0 si elle manque et que c'est pertinent
            # df[col] = 0

    # S'assurer que les codes géo sont des chaînes de caractères formatées correctement
    for col in ['CODGEO', 'REG', 'DEP']:
         if col in df.columns:
             print(f"--- Effectifs: Processing geo column '{col}'...")
             # 1. Vérifier si la colonne n'est PAS déjà de type string/object
             if not pd.api.types.is_string_dtype(df[col]):
                 print(f"--- Effectifs: Column '{col}' is not string, attempting conversion.")
                 try:
                    # Convertir en string de manière robuste, gérant NaN et types numériques
                    # fillna(-1) puis replace('-1', NA) est une astuce pour gérer les NaN pendant la conversion int->str
                    df[col] = df[col].fillna(-1).astype(int).astype(str).replace('-1', pd.NA)
                    print(f"--- Effectifs: Column '{col}' converted via int -> str.")
                 except ValueError: # Si contient des strings non numériques, convertir simplement en str
                    print(f"--- Effectifs: Column '{col}' conversion via int failed (likely mixed types), converting directly to str.")
                    df[col] = df[col].astype(str)
                 except Exception as e_conv:
                    # st.error(f"Erreur de conversion en string pour la colonne '{col}' dans Effectifs: {e_conv}")
                    print(f"Error: String conversion error for column '{col}' in Effectifs: {e_conv}")
                    # En cas d'échec grave, on peut arrêter ou continuer avec la colonne potentiellement problématique
                    # return None # Optionnel: arrêter si la colonne est critique

             # 2. Nettoyer les '.0' potentiels si la lecture initiale était float
             # S'assurer que c'est bien une string avant d'utiliser .str
             if pd.api.types.is_string_dtype(df[col]):
                 df[col] = df[col].str.replace(r'\.0$', '', regex=True)
             else: # Si ce n'est toujours pas une string après les tentatives de conversion
                  df[col] = df[col].astype(str).str.replace(r'\.0$', '', regex=True)


             # 3. Appliquer zfill pour les zéros non significatifs (uniquement sur les valeurs valides)
             zfill_len = 0
             if col in ['REG', 'DEP']:
                 zfill_len = 2
             elif col == 'CODGEO':
                 zfill_len = 5

             if zfill_len > 0:
                # Assurer que c'est string avant apply
                if not pd.api.types.is_string_dtype(df[col]):
                    df[col] = df[col].astype(str) # Dernière tentative de conversion str

                # Gérer les <NA> ou 'nan' stringifiés avant zfill
                df[col] = df[col].apply(
                    lambda x: x.zfill(zfill_len) if pd.notna(x) and x not in ['<NA>', 'nan', 'None'] else x
                )
                print(f"--- Effectifs: Column '{col}' zero-padded to {zfill_len} digits.")
         else:
              # st.warning(f"Colonne géographique attendue '{col}' manquante dans Effectifs.")
              print(f"Warning: Expected geo column '{col}' missing in Effectifs.")


    print(f"--- Effectifs data preprocessed. Shape: {df.shape} ---")
    # print("--- Effectifs dtypes after processing:\n", df.dtypes) # Pour débogage
    # print("--- Effectifs head after processing:\n", df.head()) # Pour débogage
    return df


# /!\ MODIFICATION CI-DESSOUS POUR GOOGLE DRIVE /!\
@st.cache_data
def _load_soutirage_internal(file_identifier):
    """Charge les données Soutirage. Gère les URL GDrive ou les chemins locaux. Mis en cache."""
    print(f"--- Executing _load_soutirage_internal for identifier: {file_identifier} ---")
    local_path_to_read = None
    temp_file_downloaded = False

    # Détecter si c'est une URL Google Drive
    if isinstance(file_identifier, str) and file_identifier.startswith('https://drive.google.com/'):
        # Définir un chemin de destination temporaire pour le téléchargement
        temp_soutirage_path = '/tmp/soutirage_temp_download.csv' # Utilise /tmp
        print(f"--- Identifier is a Google Drive URL. Attempting download to {temp_soutirage_path}... ---")
        try:
            # Supprimer l'ancien fichier temporaire s'il existe
            if os.path.exists(temp_soutirage_path):
                os.remove(temp_soutirage_path)
                print(f"--- Removed existing temp file: {temp_soutirage_path} ---")

            # Télécharger le fichier depuis Google Drive
            gdown.download(url=file_identifier, output=temp_soutirage_path, quiet=False, fuzzy=True) # fuzzy=True aide
            print(f"--- Google Drive file downloaded successfully to {temp_soutirage_path} ---")
            local_path_to_read = temp_soutirage_path
            temp_file_downloaded = True

        except Exception as e_gdown:
            st.error(f"Erreur lors du téléchargement depuis Google Drive ({file_identifier}): {e_gdown}")
            print(f"Error: Failed to download from Google Drive URL {file_identifier}: {e_gdown}")
            traceback.print_exc()
            return None
    else:
        # Si ce n'est pas une URL GDrive, on suppose que c'est un chemin local
        print(f"--- Identifier is treated as a local path: {file_identifier} ---")
        if not os.path.exists(file_identifier):
             print(f"Error: Local file not found: {file_identifier}")
             st.error(f"Fichier local non trouvé: {file_identifier}")
             return None
        local_path_to_read = file_identifier

    # Vérifier si un chemin de lecture a été défini
    if local_path_to_read is None:
        print("Error: No valid local path determined for reading.")
        st.error("Impossible de déterminer le fichier à lire (problème de téléchargement ou chemin local invalide).")
        return None

    # --- Logique de lecture Pandas (inchangée, mais utilise local_path_to_read) ---
    df = None
    try:
        df = pd.read_csv(local_path_to_read, sep=';', encoding='utf-8')
        print(f"--- Soutirage loaded from '{local_path_to_read}' with sep=';', encoding='utf-8'.")
    except (UnicodeDecodeError, pd.errors.ParserError):
        print(f"Warning: Failed reading Soutirage from '{local_path_to_read}' with semicolon/utf-8. Trying semicolon/latin-1...")
        try:
             df = pd.read_csv(local_path_to_read, sep=';', encoding='latin-1')
             print(f"--- Soutirage loaded from '{local_path_to_read}' with sep=';', encoding='latin-1'.")
        except (UnicodeDecodeError, pd.errors.ParserError):
             print(f"Warning: Failed reading Soutirage from '{local_path_to_read}' with semicolon/latin-1. Trying comma/utf-8...")
             try:
                 df = pd.read_csv(local_path_to_read, sep=',', encoding='utf-8')
                 print(f"--- Soutirage loaded from '{local_path_to_read}' with sep=',', encoding='utf-8'.")
             except (UnicodeDecodeError, pd.errors.ParserError):
                print(f"Warning: Failed reading Soutirage from '{local_path_to_read}' with comma/utf-8. Trying comma/latin-1...")
                try:
                    df = pd.read_csv(local_path_to_read, sep=',', encoding='latin-1')
                    print(f"--- Soutirage loaded from '{local_path_to_read}' with sep=',', encoding='latin-1'.")
                except Exception as final_e:
                    print(f"Error: Final read error for Soutirage from '{local_path_to_read}': {final_e}")
                    st.error(f"Erreur finale de lecture Soutirage {local_path_to_read}: {final_e}")
                    traceback.print_exc()
                    return None
             except Exception as e_comma:
                 print(f"Error: Read error for Soutirage from '{local_path_to_read}' with sep=',': {e_comma}")
                 st.error(f"Erreur de lecture Soutirage {local_path_to_read} avec sep=',': {e_comma}")
                 traceback.print_exc()
                 return None
        except Exception as e_semicolon:
            print(f"Error: Read error for Soutirage from '{local_path_to_read}' with sep=';': {e_semicolon}")
            st.error(f"Erreur de lecture Soutirage {local_path_to_read} avec sep=';': {e_semicolon}")
            traceback.print_exc()
            return None
    except pd.errors.EmptyDataError:
         print(f"Warning: Soutirage file {local_path_to_read} is empty.")
         st.warning(f"Le fichier Soutirage {local_path_to_read} est vide.")
         df = pd.DataFrame()
    except FileNotFoundError:
        print(f"ERROR: File {local_path_to_read} not found unexpectedly.")
        st.error(f"ERREUR: Fichier non trouvé de manière inattendue à {local_path_to_read}")
        return None
    except Exception as e:
         print(f"Error: Unexpected error loading Soutirage from '{local_path_to_read}': {e}")
         st.error(f"Erreur inattendue lors du chargement Soutirage ({local_path_to_read}): {e}")
         traceback.print_exc()
         return None

    # Nettoyage du fichier temporaire SI il a été téléchargé
    # if temp_file_downloaded and os.path.exists(local_path_to_read):
    #     try:
    #         os.remove(local_path_to_read)
    #         print(f"--- Cleaned up temporary file: {local_path_to_read} ---")
    #     except Exception as e_clean:
    #         print(f"Warning: Could not clean up temporary file {local_path_to_read}: {e_clean}")

    if df is None and not isinstance(df, pd.DataFrame):
        print(f"Error: Cannot read Soutirage file {local_path_to_read} with tested separators/encodings.")
        st.error(f"Impossible de lire le fichier Soutirage {local_path_to_read} avec les séparateurs/encodages testés.")
        return None

    # Ajouter ici un pré-traitement spécifique si nécessaire (dates, numériques...)
    # if not df.empty:
    #    Exemple:
    #    if 'Date' in df.columns:
    #        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    #    if 'Soutirage (MW)' in df.columns: # Adapter le nom de colonne
    #        df['Soutirage (MW)'] = pd.to_numeric(df['Soutirage (MW)'], errors='coerce')

    print(f"--- Soutirage loaded/processed. Shape: {df.shape} ---")
    return df
# /!\ FIN MODIFICATION SOUTIRAGE /!\

@st.cache_data
def _load_temperature_internal(file_path):
    """Charge et prépare les données Température. Mis en cache."""
    # st.toast(f"⏳ Chargement Température depuis {file_path}...") # Moins de toasts
    print(f"--- Executing _load_temperature_internal for {file_path} ---")
    if not os.path.exists(file_path):
        # st.error(f"Fichier Température non trouvé: {file_path}")
        print(f"Error: Temperature file not found: {file_path}")
        return None
    df = None
    try:
        df = pd.read_csv(file_path, sep=';', encoding='utf-8')
        print("--- Temperature loaded successfully with sep=';' and encoding='utf-8'.")
    except (UnicodeDecodeError, pd.errors.ParserError):
        print("Warning: Failed reading Temperature with semicolon/utf-8. Trying semicolon/latin-1...")
        try:
            df = pd.read_csv(file_path, sep=';', encoding='latin-1')
            print("--- Temperature loaded successfully with sep=';' and encoding='latin-1'.")
        except (UnicodeDecodeError, pd.errors.ParserError):
            print("Warning: Failed reading Temperature with semicolon/latin-1. Trying comma/utf-8...")
            # st.warning(f"Échec lecture Température avec sep=';'. Essai avec sep=','...")
            try:
                df = pd.read_csv(file_path, sep=',', encoding='utf-8')
                print("--- Temperature loaded successfully with sep=',', encoding='utf-8'.")
            except (UnicodeDecodeError, pd.errors.ParserError):
                print("Warning: Failed reading Temperature with comma/utf-8. Trying comma/latin-1...")
                try:
                    df = pd.read_csv(file_path, sep=',', encoding='latin-1')
                    print("--- Temperature loaded successfully with sep=',', encoding='latin-1'.")
                except Exception as final_e:
                    # st.error(f"Erreur finale de lecture Température {file_path}: {final_e}")
                    print(f"Error: Final read error for Temperature {file_path}: {final_e}")
                    traceback.print_exc()
                    return None
            except Exception as e_comma:
                 # st.error(f"Erreur de lecture Température {file_path} avec sep=',': {e_comma}")
                 print(f"Error: Read error for Temperature {file_path} with sep=',': {e_comma}")
                 traceback.print_exc()
                 return None
        except Exception as e_semicolon:
            # st.error(f"Erreur de lecture Température {file_path} avec sep=';': {e_semicolon}")
            print(f"Error: Read error for Temperature {file_path} with sep=';': {e_semicolon}")
            traceback.print_exc()
            return None
    except pd.errors.EmptyDataError:
        # st.warning(f"Le fichier Température {file_path} est vide.")
        print(f"Warning: Temperature file {file_path} is empty.")
        return pd.DataFrame()
    except FileNotFoundError:
        # st.error(f"ERREUR: Fichier Température non trouvé à {file_path}")
        print(f"ERROR: Temperature file not found at {file_path}")
        return None
    except Exception as e:
        # st.error(f"Erreur inattendue lors du chargement Température ({file_path}): {e}")
        print(f"Error: Unexpected error loading Temperature ({file_path}): {e}")
        traceback.print_exc()
        return None

    if df is None:
        # st.error(f"Impossible de lire le fichier Température {file_path} avec les séparateurs/encodages testés.")
        print(f"Error: Cannot read Temperature file {file_path} with tested separators/encodings.")
        return None

    if df.empty:
        # st.warning(f"Le fichier Température {file_path} est vide après lecture.")
        print(f"Warning: Temperature file {file_path} is empty after read.")
        return df

    # --- Preprocessing ---
    if 'Date' in df.columns:
        # Essayer plusieurs formats ou laisser pandas déduire, puis vérifier
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True) # dayfirst=True aide pour formats DD/MM/YYYY
        # Supprimer les lignes où la date est invalide (NaT)
        initial_rows = len(df)
        df.dropna(subset=['Date'], inplace=True)
        if len(df) < initial_rows:
             print(f"--- Temperature: Removed {initial_rows - len(df)} rows with invalid dates.")
        if df.empty:
            # st.error("Aucune date valide trouvée dans le fichier Température après conversion.")
            print("Error: No valid dates found in Temperature file after conversion.")
            return pd.DataFrame()
        print("--- Temperature: 'Date' column converted to datetime.")
    else:
        # st.error("Colonne 'Date' manquante dans le fichier Température.")
        print("Error: 'Date' column missing in Temperature file.")
        return pd.DataFrame() # Retourner DF vide car la date est essentielle

    temp_cols = ['TMin (°C)', 'TMax (°C)', 'TMoy (°C)']
    for col in temp_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"--- Temperature: Column '{col}' converted to numeric.")
        else:
            # st.warning(f"Colonne de température attendue '{col}' manquante.")
            print(f"Warning: Expected temperature column '{col}' missing.")

    if 'Région' not in df.columns:
        # st.error("Colonne 'Région' manquante, nécessaire pour le filtrage et l'analyse.")
        print("Error: 'Région' column missing, required for filtering and analysis.")
        # Selon l'importance, retourner None ou un DF vide
        return pd.DataFrame()

    print(f"--- Temperature data preprocessed. Shape: {df.shape} ---")
    return df


@st.cache_data
def _load_population_internal(file_path):
    """Charge et transforme les données Population. Mis en cache."""
    # st.toast(f"⏳ Chargement Population depuis {file_path}...") # Moins de toasts
    print(f"--- Executing _load_population_internal for {file_path} ---")
    if not os.path.exists(file_path):
         # st.error(f"Fichier Population non trouvé: {file_path}")
         print(f"Error: Population file not found: {file_path}")
         return None
    df = None
    try:
        # Tenter lecture avec virgule d'abord
        df = pd.read_csv(file_path, sep=',', encoding='utf-8')
        print("--- Population loaded with sep=',' and encoding='utf-8'.")
    except (UnicodeDecodeError, pd.errors.ParserError):
        print("Warning: Failed reading Population with comma/utf-8. Trying comma/latin-1...")
        try:
            df = pd.read_csv(file_path, sep=',', encoding='latin-1')
            print("--- Population loaded with sep=',' and encoding='latin-1'.")
        except (UnicodeDecodeError, pd.errors.ParserError):
            print("Warning: Failed reading Population with comma/latin-1. Trying semicolon/utf-8...")
            # st.warning(f"Échec lecture Population avec sep=',': Essai avec sep=';'.")
            try:
                df = pd.read_csv(file_path, sep=';', encoding='utf-8')
                print("--- Population loaded with sep=';' and encoding='utf-8'.")
            except (UnicodeDecodeError, pd.errors.ParserError):
                 print("Warning: Failed reading Population with semicolon/utf-8. Trying semicolon/latin-1...")
                 try:
                    df = pd.read_csv(file_path, sep=';', encoding='latin-1')
                    print("--- Population loaded with sep=';' and encoding='latin-1'.")
                 except Exception as final_e:
                     # st.error(f"Échec final lecture Population {file_path}: {final_e}")
                     print(f"Error: Final read error for Population {file_path}: {final_e}")
                     traceback.print_exc()
                     return None
            except Exception as e_semi:
                 # st.error(f"Erreur lecture Population {file_path} avec sep=';': {e_semi}")
                 print(f"Error: Read error for Population {file_path} with sep=';': {e_semi}")
                 traceback.print_exc()
                 return None
        except Exception as e_comma:
            # st.error(f"Erreur lecture Population {file_path} avec sep=',': {e_comma}")
            print(f"Error: Read error for Population {file_path} with sep=',': {e_comma}")
            traceback.print_exc()
            return None
    except pd.errors.EmptyDataError:
         # st.warning(f"Le fichier Population {file_path} est vide.")
         print(f"Warning: Population file {file_path} is empty.")
         return pd.DataFrame()
    except FileNotFoundError:
        # st.error(f"ERREUR: Fichier Population non trouvé à {file_path}")
        print(f"ERROR: Population file not found at {file_path}")
        return None
    except Exception as e:
        # st.error(f"Erreur inattendue chargement Population ({file_path}): {e}")
        print(f"Error: Unexpected error loading Population ({file_path}): {e}")
        traceback.print_exc()
        return None

    if df is None:
         # st.error(f"Impossible de lire le fichier Population {file_path} avec les séparateurs/encodages testés.")
         print(f"Error: Cannot read Population file {file_path} with tested separators/encodings.")
         return None

    if df.empty:
         # st.warning(f"Fichier Population {file_path} vide après lecture.")
         print(f"Warning: Population file {file_path} empty after read.")
         return df

    # --- Gestion de la colonne Date ---
    date_col_found = False
    if 'Date' in df.columns:
        print("--- Population: 'Date' column found.")
        date_col_found = True
    else:
        # Essayer si la première colonne est une date
        first_col_name = df.columns[0]
        print(f"--- Population: No 'Date' column. Checking first column '{first_col_name}'...")
        try:
            # Essayer de convertir sans modifier le df pour juste vérifier
            pd.to_datetime(df[first_col_name], errors='raise', dayfirst=True)
            df.rename(columns={first_col_name: 'Date'}, inplace=True)
            print(f"--- Population: Renamed first column '{first_col_name}' to 'Date'.")
            date_col_found = True
        except (ValueError, TypeError, KeyError, IndexError):
            # st.error("La colonne 'Date' est manquante et la première colonne ne semble pas contenir de dates valides.")
            print("Error: 'Date' column missing and first column does not appear to contain valid dates.")
            traceback.print_exc()
            return pd.DataFrame() # Date est essentielle

    if not date_col_found or 'Date' not in df.columns:
         # st.error("Erreur critique : Impossible d'identifier ou de créer la colonne 'Date'.")
         print("Critical Error: Cannot identify or create 'Date' column.")
         return pd.DataFrame()

    # --- Conversion de la colonne Date ---
    try:
        # Essayer format DD/MM/YYYY explicitement
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='raise')
        print("--- Population: Date parsed with format %d/%m/%Y.")
    except ValueError:
        print("Warning: Parsing with %d/%m/%Y failed. Trying %Y-%m-%d...")
        try:
             # Essayer format YYYY-MM-DD explicitement
             df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='raise')
             print("--- Population: Date parsed with format %Y-%m-%d.")
        except ValueError:
            print("Warning: Specific date formats failed. Using generic parser with dayfirst=True...")
            # Tentative plus générale, utile pour DD/MM/YY ou autres variations
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
            print("--- Population: Date parsed with generic parser.")

    # Supprimer les lignes où la date n'a pas pu être convertie
    initial_rows_pop = len(df)
    df.dropna(subset=['Date'], inplace=True)
    if len(df) < initial_rows_pop:
         print(f"--- Population: Removed {initial_rows_pop - len(df)} rows with invalid dates after conversion attempts.")
    if df.empty:
         # st.error("Aucune date valide trouvée après conversion dans le fichier Population.")
         print("Error: No valid dates found after conversion in Population file.")
         return pd.DataFrame()

    # --- Transformation Wide to Long (Melt) ---
    region_cols = [col for col in df.columns if col != 'Date']
    if not region_cols:
         # st.error("Aucune colonne de région/valeur trouvée pour la transformation (melt).")
         print("Error: No region/value columns found for melt transformation.")
         return pd.DataFrame()

    try:
        df_long = pd.melt(df, id_vars=['Date'], value_vars=region_cols,
                          var_name='Région', value_name='Population_raw')
        print(f"--- Population data reshaped (melted). Shape: {df_long.shape} ---")
    except Exception as e_melt:
        # st.error(f"Erreur lors du 'melt' (transformation format long) pour Population: {e_melt}")
        print(f"Error during melt (long format transformation) for Population: {e_melt}")
        traceback.print_exc()
        return None # Erreur critique

    # --- Nettoyage et Conversion des valeurs de Population ---
    if 'Population_raw' in df_long.columns:
         # 1. Convertir en string, nettoyer espaces et remplacer virgule décimale par point
         df_long['Population_cleaned'] = df_long['Population_raw'].astype(str).str.replace(' ', '', regex=False).str.replace(',', '.', regex=False)
         # 2. Convertir en numérique, les erreurs deviennent NaN
         df_long['Population'] = pd.to_numeric(df_long['Population_cleaned'], errors='coerce')
         # 3. Supprimer les lignes où la population est invalide (NaN)
         initial_rows_long = len(df_long)
         df_long.dropna(subset=['Population'], inplace=True)
         if len(df_long) < initial_rows_long:
              print(f"--- Population: Removed {initial_rows_long - len(df_long)} rows with invalid population values.")
    else:
        # st.error("Colonne 'Population_raw' non créée par melt. Impossible de traiter les valeurs.")
        print("Error: 'Population_raw' column not created by melt. Cannot process values.")
        return pd.DataFrame()

    if df_long.empty:
        # st.error("DataFrame Population vide après conversion/nettoyage des valeurs de population.")
        print("Error: Population DataFrame empty after converting/cleaning population values.")
        return pd.DataFrame()

    # 4. Convertir en entier (si approprié pour la population)
    try:
        df_long['Population'] = df_long['Population'].astype(int)
    except ValueError:
        # st.warning("Impossible de convertir toutes les valeurs de Population en entier. Elles resteront en flottant.")
        print("Warning: Cannot convert all Population values to integer. They will remain float.")
        # Garder en float si la conversion échoue
        pass

    # 5. Supprimer les colonnes intermédiaires
    df_long.drop(columns=['Population_raw', 'Population_cleaned'], inplace=True, errors='ignore')

    print(f"--- Population data fully processed (long format). Final Shape: {df_long.shape} ---")
    return df_long


# /!\ AJOUT FONCTION LOAD DF_FINAL /!\
@st.cache_data
def _load_df_final_internal(file_path):
    """Charge et prépare df_final. Mis en cache."""
    print(f"--- Executing _load_df_final_internal for {file_path} ---")
    if not os.path.exists(file_path):
        print(f"Error: df_final file not found: {file_path}")
        # Ne pas utiliser st.error ici, la fonction appelante le fera
        return None
    df = None
    try:
        # Essayer avec la virgule en premier, basé sur l'aperçu fourni
        df = pd.read_csv(file_path, sep=',', encoding='utf-8')
        print("--- df_final loaded successfully with sep=',' and encoding='utf-8'.")
    except Exception as e_comma:
        print(f"Warning: Failed reading df_final with sep=',': {e_comma}. Trying sep=';'...")
        try:
            df = pd.read_csv(file_path, sep=';', encoding='utf-8') # Essayer ;/utf-8
            print("--- df_final loaded successfully with sep=';' and encoding='utf-8'.")
        except Exception as e_semi:
            print(f"Error: Failed reading df_final with both sep=',' and sep=';': {e_semi}")
            traceback.print_exc()
            return None # Échec du chargement

    if df is None:
        print(f"Error: Cannot read df_final file {file_path} with tested separators/encodings.")
        return None

    if df.empty:
        print(f"Warning: df_final file {file_path} is empty after read.")
        return df

    # --- Prétraitement spécifique à df_final ---
    # Convertir la date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True) # Important
        # Extraire année et mois pour faciliter les calculs
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        print("--- df_final: 'date' column processed, 'year' and 'month' extracted.")
    else:
        print("Error: 'date' column missing in df_final. Contextual info will fail.")
        return pd.DataFrame() # Date est essentielle

    # Assurer que les colonnes numériques sont bien numériques
    num_cols_context = ['population', 'nb_total_entreprise', 'tmoy_degc']
    for col in num_cols_context:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"--- df_final: Column '{col}' converted to numeric.")
        else:
             print(f"Warning: Expected context column '{col}' missing in df_final.")
             # Créer la colonne avec NaN si elle manque ? Ou laisser tel quel. Laisser tel quel pour l'instant.

    # Correction de l'encodage potentiel du nom de région (copié depuis train_model.py)
    if 'region' in df.columns:
         df['region'] = df['region'].astype(str).str.replace('Ã´', 'ô').str.replace('Ã¨', 'è').str.replace('Ã©', 'é').str.replace('Ã§', 'ç')
         print("--- df_final: Region names potentially corrected for encoding issues.")

    print(f"--- df_final data preprocessed. Shape: {df.shape} ---")
    return df

# =============================================================================
# --- 5. GETTER FUNCTIONS (Lazy Loading Logic using st.session_state) ---
# =============================================================================
# Ces fonctions sont l'interface publique pour accéder aux données.
# Elles utilisent les fonctions internes (_) pour charger si nécessaire.

def get_eco2mix_data():
    """Récupère les données Eco2mix (charge si nécessaire via la fonction interne)."""
    if 'eco2mix_df' not in st.session_state:
        print("--- Eco2mix data not in session state. Calling internal loader. ---")
        # /!\ CORRECTION: Utilise la variable définie dans la section 1 /!\
        st.session_state.eco2mix_df = _load_eco2mix_internal(ECO2MIX_CSV_PATH)
    # Toujours retourner une copie pour éviter la modification de l'état en cache
    if st.session_state.eco2mix_df is not None:
        return st.session_state.eco2mix_df.copy()
    else:
        return None

def get_effectifs_data():
    """Récupère les données Effectifs (charge si nécessaire via la fonction interne)."""
    if 'effectifs_df' not in st.session_state:
        print("--- Effectifs data not in session state. Calling internal loader. ---")
        # /!\ CORRECTION: Utilise la variable définie dans la section 1 /!\
        st.session_state.effectifs_df = _load_effectifs_internal(EFFECTIFS_CSV_PATH)
    if st.session_state.effectifs_df is not None:
        return st.session_state.effectifs_df.copy()
    else:
        return None

def get_soutirage_data():
    """Récupère les données Soutirage (charge si nécessaire via la fonction interne)."""
    if 'soutirage_df' not in st.session_state:
        print("--- Soutirage data not in session state. Calling internal loader. ---")
        # /!\ CORRECTION: Utilise la variable définie dans la section 1 /!\
        st.session_state.soutirage_df = _load_soutirage_internal(SOUTIRAGE_CSV_PATH)
    if st.session_state.soutirage_df is not None:
        return st.session_state.soutirage_df.copy()
    else:
        return None

def get_temperature_data():
    """Récupère les données Température (charge si nécessaire via la fonction interne)."""
    if 'temperature_df' not in st.session_state:
        print("--- Temperature data not in session state. Calling internal loader. ---")
        # /!\ CORRECTION: Utilise la variable définie dans la section 1 /!\
        st.session_state.temperature_df = _load_temperature_internal(TEMPERATURE_CSV_PATH)
    if st.session_state.temperature_df is not None:
        return st.session_state.temperature_df.copy()
    else:
        return None

def get_population_data():
    """Récupère les données Population (format long) (charge si nécessaire via la fonction interne)."""
    if 'population_df_long' not in st.session_state:
        print("--- Population data (long format) not in session state. Calling internal loader. ---")
        # /!\ CORRECTION: Utilise la variable définie dans la section 1 /!\
        st.session_state.population_df_long = _load_population_internal(POPULATION_CSV_PATH)
    if st.session_state.population_df_long is not None:
        return st.session_state.population_df_long.copy()
    else:
        return None


# /!\ AJOUT GETTER DF_FINAL /!\
def get_df_final_data():
    """Récupère les données df_final (charge si nécessaire via la fonction interne)."""
    if 'df_final_data' not in st.session_state:
        print("--- df_final data not in session state. Calling internal loader. ---")
        # Utilise la variable définie dans la section 1 (qui était déjà correcte)
        st.session_state.df_final_data = _load_df_final_internal(DF_FINAL_CSV_PATH)
    # Toujours retourner une copie pour éviter la modification de l'état en cache
    if st.session_state.df_final_data is not None:
        return st.session_state.df_final_data.copy()
    else:
        return None


# =============================================================================
# --- 6. VISUALIZATIONS DICTIONARY ---
# =============================================================================
# (Code inchangé ici)
visualizations_data = {
    "1. Évolution de la consommation d'énergie en France (2013-2023)": {
        "category": "Consommation (Tendances & Saisons)", # Ancienne catégorie
        "text": """Le graphique ci-dessus illustre l'évolution de la consommation d'énergie globale (en MW) sur une
période proche de la décennie. On observe des variations saisonnières marquées, avec des pics de
consommation réguliers correspondant probablement aux périodes hivernales, où la demande en
chauffage augmente considérablement. Ces cycles montrent une tendance annuelle répétitive,
indiquant que la consommation énergétique est influencée de manière significative par les conditions
climatiques et les habitudes de consommation. Les données mettent également en évidence une
certaine stabilité dans les tendances générales de la consommation d'énergie au fil du temps."""
    },
    "2. Production et consommation électrique : Défis de 2022": {
        "category": "Production & Mix Énergétique", # Ancienne catégorie
        "text": """Le graphique met en évidence la comparaison entre la consommation d'électricité (en rouge) et la
production totale d'électricité (en bleu) en France pour l'année 2022, avec des périodes de déficit de
production illustrées par les zones en rose. Notamment, la consommation a souvent dépassé la
production, en particulier lors des mois hivernaux, ce qui peut être attribué à une demande accrue en
chauffage. Cette situation a été exacerbée par l'indisponibilité de plusieurs réacteurs nucléaires
français en 2022, due à des problèmes de corrosion sous contrainte détectés sur certaines
tuyauteries, entraînant des arrêts prolongés pour maintenance et réparations.

Ces indisponibilités ont conduit la France, habituellement exportatrice nette d'électricité, à importer de
l'électricité pour répondre à la demande intérieure.

Ce visuel souligne l'importance cruciale de la gestion et de la maintenance des infrastructures de
production énergétique pour assurer l'équilibre entre l'offre et la demande, particulièrement en période
de pointe.

**Références:**
- [Banque de France - Solde énergétique en 2022](https://www.banque-france.fr/fr/publications-et-statistiques/publications/solde-energetique-en-2022-la-crise-de-la-production-electronucleaire-survenue-au-pire-moment)
- [Primeo Energie - État des lieux du parc nucléaire français](https://www.primeo-energie.fr/actuenergie/etat-des-lieux-du-parc-nucleaire-francais/)"""
    },
    "3. Répartition régionale de la consommation totale d'énergie en France (2013 à 2023)": {
        "category": "Analyses Régionales (Énergie & Population)", # Ancienne catégorie
        "text": """Ce graphique circulaire présente la répartition de la consommation d'énergie par région en France.
On constate que l'Île-de-France représente la part la plus importante, avec 15 % de la consommation
totale, ce qui peut s'expliquer par la densité de population et la concentration d'activités économiques
dans cette région. Elle est suivie par la région Auvergne-Rhône-Alpes (14 %), qui inclut de grandes
villes industrielles comme Lyon, et par le Grand Est (10,7 %), connu pour ses besoins énergétiques
élevés dans le secteur industriel. À l'inverse, des régions comme le Centre-Val de Loire et la
Bourgogne-Franche-Comté affichent les parts les plus faibles, avec respectivement 4 % et 4,5 %. Ce
visuel met en lumière les disparités régionales de la consommation énergétique, qui sont influencées
par la démographie, l'industrialisation et les conditions climatiques locales."""
    },
    "4. Évolution de la production d'énergie par source (2013-2022)": {
        "category": "Production & Mix Énergétique", # Ancienne catégorie
        "text": """Ce graphique en barres empilées présente l'évolution de la répartition des différentes sources
d'énergie en France de 2013 à 2022. La production nucléaire (en gris) constitue la majeure partie de
la production énergétique chaque année, confirmant le rôle prédominant de cette source dans le mix
énergétique français. Cependant, on observe une légère diminution de la part du nucléaire au cours
des dernières années, notamment en 2022, en raison des indisponibilités prolongées de plusieurs
réacteurs nucléaires pour des maintenances et réparations.

Les sources renouvelables, telles que l'éolien (en vert) et le solaire (en jaune), montrent une
croissance progressive au fil des années, bien que leur part reste encore limitée par rapport au
nucléaire et au thermique. L'hydraulique (en bleu) reste une source stable, mais dépend fortement
des conditions climatiques. Le thermique (en rouge), quant à lui, joue un rôle de soutien pour
compenser les fluctuations des autres sources, notamment en période de forte demande ou de
déficience du parc nucléaire.

Ce visuel met en évidence la transition énergétique progressive en France, marquée par une
diversification des sources d'énergie et une montée en puissance des énergies renouvelables, bien
que le nucléaire reste un pilier central du système énergétique français."""
    },
    "5. Carte de chaleur de la consommation mensuelle d'électricité (2013-2023)": {
        "category": "Consommation (Tendances & Saisons)", # Ancienne catégorie
        "text": """La carte de chaleur mensuelle de la consommation électrique met en évidence les variations
saisonnières de la demande énergétique en France, réparties par année et par mois. On observe des
pics de consommation récurrents en hiver, notamment en janvier et décembre, marqués par des
teintes rouges foncées, ce qui correspond aux périodes de forte demande liée au chauffage. À
l'inverse, les mois d'été, en particulier de mai à septembre, montrent une consommation nettement
plus faible, représentée par des teintes bleues.

L'année 2017 se distingue par un pic exceptionnel de consommation en janvier, probablement en
raison de conditions climatiques extrêmes, comme une vague de froid. Les tendances générales
montrent une cyclicité annuelle stable, avec des hausses hivernales et des baisses estivales.

Ce visuel permet de mieux comprendre la relation entre les conditions climatiques et la demande en
électricité, soulignant l'importance d'une planification énergétique efficace pour répondre aux besoins
accrus en période hivernale. Il met également en avant l'impact des aléas climatiques sur les
variations exceptionnelles de la consommation électrique."""
    },
    "6. Distribution de la consommation électrique moyenne par tranche de demie-heure par saison au niveau national": {
        "category": "Consommation (Tendances & Saisons)", # Ancienne catégorie
        "text": """Le graphique en boîtes à moustaches (boxplot) présente la distribution de la consommation d'énergie
en France selon les saisons. On observe que la consommation est nettement plus élevée en hiver,
avec une médiane située autour de 6 000 MW et des valeurs maximales atteignant près de 16 000
MW, en raison de la forte demande liée au chauffage. L'automne suit avec une consommation
relativement élevée, tandis que le printemps et l'été affichent des niveaux de consommation plus
faibles.

Les boîtes à moustaches montrent également une plus grande variabilité en hiver, avec de nombreux
points au-dessus des moustaches, indiquant des valeurs extrêmes (pics de consommation). En
revanche, les saisons plus chaudes (printemps et été) présentent des distributions plus homogènes,
avec moins de valeurs extrêmes.

Ce visuel met en évidence l'impact des conditions climatiques sur la demande en électricité,
soulignant l'importance de la saisonnalité dans la gestion de la production et des infrastructures
énergétiques. La forte consommation hivernale rappelle également la nécessité d'anticiper les
périodes de forte demande pour éviter les tensions sur le réseau électrique."""
    },
    "7. Évolution temporelle par jour de la consommation électrique moyenne par saison": {
        "category": "Consommation (Tendances & Saisons)", # Ancienne catégorie
        "text": """Le graphique montre l'évolution temporelle de la consommation moyenne d'électricité par saison,
répartie tout au long de l'année. La courbe met en évidence des variations saisonnières bien
distinctes. En hiver (en orange), la consommation d'électricité atteint ses plus hauts niveaux,
dépassant 6 000 MW en raison des besoins accrus de chauffage. À l'inverse, durant l'été (en rouge),
la consommation est au plus bas, avec une moyenne autour de 3 500 MW, ce qui reflète une moindre
utilisation de chauffage et une consommation globalement plus stable.

Le printemps (en vert) et l'automne (en bleu) affichent des niveaux intermédiaires, mais la transition
entre les saisons montre une tendance claire : la consommation augmente fortement à l'approche de
l'hiver et diminue progressivement après cette période. Les hausses et baisses sont régulières et
suivent les cycles naturels des variations climatiques.

Ce visuel met en évidence la forte corrélation entre les saisons et la consommation énergétique. Il
souligne l'importance de prévoir la demande énergétique en fonction des périodes de l'année afin
d'optimiser les capacités de production et de répondre aux besoins de manière efficace."""
    },
    "8. Variations journalières de la consommation électrique en France": {
        "category": "Consommation (Tendances & Saisons)", # Ancienne catégorie
        "text": """Le graphique illustre la distribution horaire moyenne de la consommation d'électricité sur une journée
typique. On observe une tendance claire, avec deux pics principaux de consommation : le premier en
fin de matinée, entre 10 h et 13 h, et le second en début de soirée, autour de 19 h. Ces pics peuvent
être attribués aux habitudes de la vie quotidienne, comme les activités matinales (chauffage,
préparation des repas, travail) et les besoins accrus en soirée après le retour à domicile (éclairage,
électroménagers, cuisine).

La consommation est la plus basse durant les heures nocturnes, entre 1 h et 5 h du matin, reflétant
une baisse de l'activité économique et domestique. À partir de 6 h, la demande commence à
augmenter progressivement jusqu'à atteindre le pic de la fin de matinée.

Ce visuel met en évidence les variations de la demande d'électricité en fonction des moments de la
journée, soulignant l'importance d'ajuster la production énergétique pour répondre aux besoins
spécifiques de ces périodes de forte consommation. Cette information est essentielle pour la gestion
des réseaux électriques et l'optimisation des infrastructures énergétiques."""
    },
    "9. Proportion de la production d'électricité en France des énergies renouvelables et non-renouvelable": {
        "category": "Production & Mix Énergétique", # Ancienne catégorie
        "text": """Ce graphique compare la production moyenne d'électricité en France entre les sources renouvelables
et non-renouvelables. Les sources non-renouvelables dominent largement, représentant 75,9 % de la
production totale, tandis que les énergies renouvelables contribuent à hauteur de 24,1 %.

La prédominance des énergies non-renouvelables s'explique principalement par la forte dépendance
au nucléaire en France, qui constitue une part importante de la production non-renouvelable. En
revanche, les énergies renouvelables incluent des sources telles que l'hydroélectricité, l'éolien, le
solaire et les bioénergies, qui, bien qu'en progression, restent minoritaires.

Ce visuel met en évidence le défi de la transition énergétique en France. Pour atteindre les objectifs
climatiques et réduire les émissions de gaz à effet de serre, il est crucial d'accroître la part des
énergies renouvelables dans le mix énergétique. Cela nécessitera des investissements importants
dans les infrastructures renouvelables et des politiques favorisant leur développement à long terme."""
    },
    "10. Contribution des énergies renouvelables par saison": {
        "category": "Production & Mix Énergétique", # Ancienne catégorie
        "text": """Le graphique en barres empilées montre la répartition moyenne des différentes sources d'énergie
renouvelable (éolien, solaire, hydraulique et bioénergies) selon les saisons. On remarque que
l'hydraulique (en vert) constitue la part la plus importante de la production d'énergie renouvelable tout
au long de l'année, en raison de la disponibilité constante de cette ressource, notamment grâce aux
barrages et aux cours d'eau. Cependant, la production hydraulique tend à être plus élevée au
printemps, probablement en raison de la fonte des neiges et des précipitations.

La production éolienne (en violet) est également stable sur l'année, avec une légère hausse en hiver
et au printemps, périodes où les vents sont généralement plus forts. En revanche, la production
solaire (en bleu) atteint son pic durant l'été, grâce à une exposition maximale au soleil.

Enfin, les bioénergies (en jaune) représentent une contribution constante et relativement stable à la
production d'énergie renouvelable, quelle que soit la saison.

Ce visuel met en évidence la complémentarité des différentes sources d'énergie renouvelable en
fonction des saisons, soulignant l'importance de diversifier les sources de production pour assurer un
approvisionnement énergétique stable tout au long de l'année."""
    },
    "11. Répartition régionale de la consommation d'électricité": {
        "category": "Analyses Régionales (Énergie & Population)", # Ancienne catégorie
        "text": """La carte présente la consommation totale d'électricité par région en France, exprimée en mégawatts
(MW). La distribution régionale met en évidence des disparités significatives entre les différentes
régions. Les régions les plus peuplées et économiquement développées, telles que l'Île-de-France,
Auvergne-Rhône-Alpes et les Hauts-de-France, affichent les plus hauts niveaux de consommation,
représentés par les teintes les plus foncées sur la carte. Cela s'explique par une forte concentration
de population, d'activités industrielles et de services nécessitant une importante consommation
d'énergie.

À l'inverse, des régions comme la Bretagne, la Normandie ou les Pays de la Loire affichent une
consommation plus modérée, en raison de leur densité de population plus faible et d'une moindre
concentration d'activités énergivores.

Ce visuel met en évidence l'importance des facteurs démographiques et économiques dans la
répartition de la consommation électrique à l'échelle régionale. Il souligne également la nécessité
d'adapter les politiques énergétiques régionales pour répondre aux besoins spécifiques de chaque
territoire, en prenant en compte les spécificités locales en matière de production et de consommation."""
    },
    "12. Production électrique : Disparités régionales": {
        "category": "Analyses Régionales (Énergie & Population)", # Ancienne catégorie
        "text": """Cette carte montre la production totale d'électricité par région en France, exprimée en mégawatts
(MW). Les teintes de bleu indiquent les variations de production entre les régions, avec les régions
les plus productrices représentées par les teintes les plus foncées. On constate que les régions du
Grand Est, d'Auvergne-Rhône-Alpes et du Centre-Val de Loire se distinguent comme les principaux
pôles de production électrique. Cela s'explique par la présence de nombreuses centrales nucléaires
dans ces régions, qui constituent une part importante du mix énergétique français.

En revanche, des régions comme la Bretagne et la Normandie affichent une production plus faible, ce
qui peut s'expliquer par une moindre densité de sites de production électrique, notamment les
centrales thermiques et nucléaires.

Ce visuel met en lumière les disparités régionales en termes de production d'énergie et souligne
l'importance stratégique de certaines régions dans l'approvisionnement électrique national. Il révèle
également la nécessité d'adapter les infrastructures de production aux besoins spécifiques de chaque
territoire pour assurer une meilleure gestion du réseau électrique."""
    },
    "13. Croissance démographique régionale (1990-2024)": { # Note: Titre incohérent avec le texte qui parle de pop 2024 et pas de croissance
        "category": "Analyses Régionales (Énergie & Population)", # Ancienne catégorie
        "text": """Cette carte représente la population totale par région en France pour l'année 2024. Les teintes
violettes indiquent les variations de densité de population, avec les régions les plus peuplées
représentées par les couleurs les plus foncées. L'Île-de-France se distingue comme la région la plus
densément peuplée, en raison de la présence de Paris et de sa région métropolitaine. Elle est suivie
par les régions Auvergne-Rhône-Alpes, Provence-Alpes-Côte d'Azur et Occitanie, qui comptent
également des métropoles importantes telles que Lyon, Marseille et Toulouse.

Les régions moins peuplées, comme la Bretagne, la Bourgogne-Franche-Comté et la Normandie,
apparaissent dans des teintes plus claires. Ces disparités démographiques influencent directement
les besoins énergétiques de chaque région, les zones les plus densément peuplées étant
susceptibles de consommer davantage d'électricité."""
    },
    "14. Croissance de la population totale en France (1990-2024)": {
        "category": "Démographie Nationale", # Ancienne catégorie
        "text": """Ce graphique montre l'évolution de la population totale en France, exprimée en millions d'habitants,
entre 1990 et 2024. La courbe bleue représente la population totale au fil des années, tandis que la
ligne rouge pointillée indique la tendance de croissance moyenne sur la période. La population a
connu une augmentation régulière, passant d'environ 56 millions en 1990 à près de 66 millions en
2024. Cette croissance équivaut à une augmentation annuelle moyenne de 0,3 million d'habitants,
soit un taux de croissance d'environ 0,53 % par an.

Le graphique met également en évidence des périodes de croissance légèrement plus rapide dans
les années 1990 et au début des années 2000. La croissance semble cependant devenir plus
modérée ces dernières années, ce qui pourrait s'expliquer par des facteurs tels que la diminution des
taux de natalité ou les politiques migratoires.

Ce visuel souligne la tendance démographique à long terme en France, qui a des implications
importantes pour la planification des infrastructures et des services publics, y compris la
consommation énergétique. La hausse constante de la population entraîne nécessairement une
augmentation de la demande en énergie et en ressources, ce qui doit être pris en compte dans les
politiques de gestion de l'énergie et de développement durable."""
    },
    "15. Répartition de la population par région en France au 31 décembre 2024": {
        "category": "Analyses Régionales (Énergie & Population)", # Ancienne catégorie
        "text": """Ce graphique en barres horizontales présente la population estimée par région en France au 31
décembre 2024, exprimée en millions d'habitants. La région Île-de-France se distingue nettement
avec plus de 12 millions d'habitants, confirmant son rôle de pôle démographique majeur. Viennent
ensuite les régions Auvergne-Rhône-Alpes et Nouvelle-Aquitaine, qui comptent respectivement
environ 8 et 6 millions d'habitants. Ces régions regroupent de grandes métropoles telles que Lyon,
Bordeaux et Toulouse, contribuant ainsi à leur densité de population.

À l'autre extrémité du spectre, la Corse est la région la moins peuplée avec environ 0,3 million
d'habitants. Les régions telles que le Centre-Val de Loire, la Bourgogne-Franche-Comté, la
Normandie et la Bretagne présentent des populations relativement stables et moins concentrées par
rapport aux grandes régions métropolitaines.

Ce visuel met en évidence les disparités régionales en termes de population, qui influencent
directement les besoins en infrastructures, en services publics et en ressources énergétiques. Les
régions les plus peuplées sont celles qui nécessitent le plus d'énergie pour alimenter les ménages,
les industries et les services. Ces informations sont cruciaux pour adapter les politiques énergétiques
aux besoins spécifiques de chaque région."""
    },
    "16. Croissance annuelle moyenne de la population par région (1990-2024)": {
        "category": "Analyses Régionales (Énergie & Population)", # Ancienne catégorie
        "text": """Ce graphique en barres horizontales montre la croissance annuelle moyenne de la population par
région en France entre 1990 et 2024. L'Île-de-France enregistre la plus forte croissance moyenne,
dépassant les 50 000 habitants par an, en raison de son attractivité économique et de son rôle de
pôle central d'activités. Suivent les régions Occitanie, Auvergne-Rhône-Alpes et Nouvelle-Aquitaine,
qui connaissent également une forte croissance démographique, attirant de nouveaux habitants
grâce à leur qualité de vie et à leurs dynamiques économiques.

Les régions ayant une croissance moyenne plus modérée incluent la Bourgogne-Franche-Comté, la
Corse et la Normandie. Cela peut s'expliquer par des facteurs comme une moindre attractivité
économique ou un vieillissement de la population.

Ce visuel met en évidence les différences régionales en termes de croissance démographique, qui
influencent directement la planification urbaine, les infrastructures et les politiques publiques. Les
régions à forte croissance devront faire face à des défis en matière d'aménagement du territoire, de
logement, et de gestion des ressources, notamment énergétiques."""
    },
    "17. Évolution de la température moyenne mensuelle en France (2016-2025)": {
        "category": "Climat & Impact Énergie", # Ancienne catégorie
        "text": """Le graphique présente l'évolution de la température moyenne mensuelle sur plusieurs années,
accompagnée d'une tendance générale de régression linéaire représentée par la ligne rouge. La
courbe bleue met en évidence les fluctuations saisonnières typiques de la température, avec des pics
élevés en été et des creux en hiver. Cependant, la ligne de tendance montre une augmentation
progressive des températures moyennes au fil des ans.

La variation annuelle moyenne indiquée est de 0,218 °C, soit une hausse de 3,25 % par an. Cette
tendance à la hausse reflète un réchauffement climatique global, avec une augmentation constante
des températures moyennes sur la période observée. Ce phénomène est cohérent avec les
préoccupations liées au changement climatique, qui impacte de nombreux aspects, notamment la
consommation énergétique pour le chauffage et la climatisation.

Ce visuel souligne l'importance d'intégrer les prévisions climatiques dans la gestion des ressources
énergétiques. Une hausse continue des températures peut modifier les besoins énergétiques
saisonniers, avec une probable augmentation de la demande estivale en climatisation et une
diminution des besoins hivernaux en chauffage."""
    },
    "18. Carte des températures moyennes annuelles par région en France": {
        "category": "Analyses Régionales (Énergie & Population)", # Ancienne catégorie
        "text": """Cette carte montre la température moyenne annuelle par région en France. Les régions du nord et du
centre du pays, comme les Hauts-de-France, la Normandie, le Grand Est, et l'Île-de-France,
présentent les températures moyennes les plus basses, représentées par les teintes bleues. À
l'opposé, les régions du sud, telles que la Provence-Alpes-Côte d'Azur, l'Occitanie et surtout la Corse,
affichent des températures plus élevées, avec des teintes allant vers le rouge.

La Corse se distingue comme la région ayant la température moyenne la plus élevée, dépassant les
16 °C. Ce contraste nord-sud est caractéristique du climat français, avec des températures plus
froides dans les régions septentrionales et plus chaudes dans les régions méditerranéennes.

Ce visuel met en évidence les variations climatiques régionales, qui influencent les besoins
énergétiques locaux. Les régions plus froides nécessitent davantage de chauffage en hiver, tandis
que les régions plus chaudes peuvent avoir une plus forte demande en climatisation durant les mois
estivaux. Ces disparités doivent être prises en compte dans la planification des infrastructures
énergétiques pour adapter les ressources aux spécificités climatiques locales."""
    },
    "19. Carte de chaleur des températures moyennes par région et mois en France": {
        "category": "Analyses Régionales (Énergie & Population)", # Ancienne catégorie
        "text": """Cette carte de chaleur illustre les températures moyennes mensuelles par région en France,
permettant de visualiser les variations saisonnières et géographiques tout au long de l'année. Les
régions méridionales, comme la Corse et la Provence-Alpes-Côte d'Azur, se démarquent par des
températures plus élevées, notamment durant les mois d'été (juin, juillet, août), où elles atteignent
jusqu'à 25 °C en moyenne. En revanche, les régions du nord et du centre, telles que les
Hauts-de-France, le Grand Est et la Bourgogne-Franche-Comté, affichent des températures
moyennes plus basses, particulièrement en hiver (janvier et février), avec des valeurs inférieures à 5 °C.

Les différences saisonnières sont très marquées, avec une hausse significative des températures
entre le printemps et l'été, suivie d'une baisse progressive à l'approche de l'automne et de l'hiver. Ce
phénomène est particulièrement visible en Corse, qui connaît les températures les plus élevées en
été, dépassant les 25 °C.

Ce visuel met en évidence la forte influence des saisons sur les variations de température en France.
Ces différences doivent être prises en compte dans les stratégies énergétiques régionales, car elles
affectent les besoins de chauffage en hiver et de climatisation en été. La carte permet également
d'identifier les régions les plus susceptibles de faire face à des vagues de chaleur, notamment dans le
sud du pays, ce qui peut avoir un impact sur la demande énergétique et les infrastructures."""
    },
    "20. Répartition des températures moyennes en France : Histogramme et courbe de distribution": {
        "category": "Climat & Impact Énergie", # Ancienne catégorie
        "text": """Le graphique représente la répartition des températures moyennes en France sous forme
d'histogramme, accompagné d'une courbe de densité. La distribution des températures suit une
forme en cloche, proche d'une distribution normale, avec une fréquence maximale autour de 10 à 15
°C. Cela indique que la majorité des températures moyennes observées en France se situent dans
cette fourchette.

Les températures plus extrêmes, inférieures à 0 °C ou supérieures à 25 °C, sont beaucoup moins
fréquentes, ce qui est cohérent avec le climat tempéré de la France. On observe une légère
asymétrie vers la droite, ce qui suggère qu'il y a une proportion légèrement plus élevée de
températures moyennes élevées par rapport aux températures basses.

Ce visuel permet de mieux comprendre les conditions climatiques générales en France, en mettant
en évidence que la majorité des températures moyennes sont modérées. Cela a des implications
importantes pour les besoins énergétiques saisonniers, notamment en matière de chauffage en hiver
et de climatisation en été, les périodes de températures extrêmes étant plus rares."""
    },
    "21. Évolution des températures moyennes annuelles en France (2016-2024)": {
        "category": "Climat & Impact Énergie", # Ancienne catégorie
        "text": """Le graphique montre l'évolution des températures moyennes annuelles en France sur une période
allant de 2016 à 2024. La courbe rouge lissée met en évidence les fluctuations interannuelles des
températures, tandis que les points noirs représentent les données réelles pour chaque année. On
observe une tendance générale à la hausse des températures, bien que cette tendance soit marquée
par des cycles de variation.

Les années 2020 et 2021 montrent une légère baisse des températures moyennes, mais cette baisse
est suivie d'une remontée notable à partir de 2022, culminant en 2024. Ces variations peuvent être
liées à des événements climatiques spécifiques ou à des phénomènes météorologiques ponctuels.

Ce visuel souligne l'importance de suivre les tendances climatiques à long terme pour mieux
comprendre l'impact du changement climatique. Bien que les variations annuelles puissent masquer
la tendance générale, l'augmentation des températures moyennes sur plusieurs années est un
indicateur clair du réchauffement climatique, avec des implications sur les besoins énergétiques, les
ressources naturelles et la gestion des infrastructures."""
    },
    "22. Répartition de la consommation d'électricité par secteur d'activité économique": {
        "category": "Consommation par Secteur", # Ancienne catégorie
        "text": """Ce graphique circulaire présente la répartition de la consommation d'électricité en France par secteur
d'activité. La grande industrie domine largement la consommation avec 66 % du total, ce qui reflète le
rôle important de l'industrie lourde et des processus industriels dans la demande énergétique
nationale. Cela inclut des secteurs tels que la métallurgie, la chimie, et les raffineries, qui nécessitent
une alimentation énergétique continue et intensive.

Le secteur "Autre" représente 23,8 % de la consommation totale. Cette catégorie peut inclure les
usages résidentiels, les infrastructures publiques, et d'autres activités moins énergivores mais
toujours essentielles, comme les transports et l'agriculture.

Le secteur tertiaire, qui regroupe les bureaux, les commerces et les services, représente 10,2 % de la
consommation totale. Bien que moins gourmand en énergie que la grande industrie, ce secteur est
néanmoins important, notamment dans les régions fortement urbanisées.

Ce visuel met en évidence l'impact majeur du secteur industriel sur la demande énergétique en
France. Il souligne la nécessité de cibler les industries pour toute initiative visant à réduire la
consommation énergétique ou à améliorer l'efficacité énergétique. Par ailleurs, bien que le secteur
tertiaire consomme moins d'énergie, il joue un rôle important dans les villes et pourrait bénéficier
d'actions visant à promouvoir les énergies renouvelables et l'efficacité énergétique dans les bâtiments
commerciaux et administratifs."""
    },
    "23. Top 5 des régions avec le plus d'entreprises (2019)": {
        "category": "Analyses Régionales (Énergie & Population)", # Ancienne catégorie -> Devrait être Économie/Secteur ? Mais focus Régional. Gardons Régional.
        "text": """Ce graphique circulaire illustre la répartition des entreprises dans les cinq régions françaises
comptant le plus grand nombre d'entreprises. L'Île-de-France domine largement avec 36,3 % du total
des entreprises, soit plus d'un million d'entreprises, confirmant son rôle de moteur économique du
pays. Cette concentration est due à la présence de Paris, capitale économique et financière, ainsi
qu'à l'attractivité de la région pour les sièges sociaux et les start-ups.

L'Auvergne-Rhône-Alpes se classe en deuxième position avec 19 % des entreprises, suivie de la
Provence-Alpes-Côte d'Azur (16 %), de l'Occitanie (14,9 %) et de la Nouvelle-Aquitaine (13,7 %). Ces
régions disposent d'importantes métropoles économiques comme Lyon, Marseille, Toulouse et
Bordeaux, qui contribuent à la création et au développement d'entreprises.

Ce visuel met en lumière l'inégalité dans la répartition des entreprises à travers le territoire français.
Les régions les plus dynamiques économiquement concentrent une grande partie des activités
entrepreneuriales, ce qui peut avoir un impact direct sur les besoins en infrastructures, en énergie et
en services. Cette répartition reflète également les disparités économiques régionales, qui doivent
être prises en compte dans les politiques de développement territorial et économique."""
    },
    "24. Consommation d'électricité par secteur d'activité économique et par région (2023)": {
        "category": "Analyses Régionales (Énergie & Population)", # Ancienne catégorie -> Croise Régional et Secteur. Mettons dans Régional.
        "text": """Ce graphique à barres empilées montre la consommation totale d'électricité par secteur d'activité et
par région en France pour l'année 2023. La consommation est décomposée en trois principaux
secteurs : la grande industrie (en bleu), le secteur tertiaire (en vert) et les autres secteurs (en orange).

Les Hauts-de-France se distinguent comme la région ayant la plus forte consommation d'énergie,
principalement due à la grande industrie, qui représente la majeure partie de la consommation dans
cette région. Cela peut être expliqué par la présence d'industries lourdes, telles que les industries
métallurgiques et chimiques. L'Auvergne-Rhône-Alpes et la Provence-Alpes-Côte d'Azur suivent
également avec une consommation industrielle élevée, tout en ayant une contribution notable du
secteur tertiaire.

En revanche, l'Île-de-France, bien que très peuplée, présente une répartition différente de sa
consommation énergétique. La consommation dans cette région est dominée par le secteur tertiaire,
en raison de la forte concentration d'entreprises de services, de bureaux et d'activités économiques
non industrielles.

Ce visuel met en évidence les disparités régionales dans la consommation énergétique selon les
secteurs d'activité. Les régions à forte industrialisation consomment davantage d'énergie dans le
secteur industriel, tandis que les régions axées sur les services, comme l'Île-de-France, voient une
plus grande part de leur consommation énergétique provenir du secteur tertiaire. Cette répartition doit
être prise en compte pour adapter les politiques énergétiques aux besoins spécifiques de chaque
région et secteur."""
    },
    "25. Evolution de la consommation électrique par secteur d'activité économique sur une journée": {
        "category": "Consommation par Secteur", # Ancienne catégorie
        "text": """Le graphique montre l'évolution de la consommation moyenne d'électricité par secteur d'activité au
cours d'une journée typique. Trois secteurs sont représentés : la grande industrie (en orange), le
secteur tertiaire (en vert) et les autres secteurs (en bleu).

La grande industrie affiche une consommation relativement constante tout au long de la journée,
autour de 700 MW. Cela s'explique par le fonctionnement continu des processus industriels, qui
nécessitent une alimentation énergétique stable, indépendamment des heures de la journée.

Le secteur tertiaire, en revanche, présente une variation plus importante au fil de la journée. La
consommation augmente progressivement à partir de 6 h du matin, atteignant un pic autour de midi,
puis redescend en fin de journée. Cette tendance est cohérente avec les horaires d'ouverture des
bureaux et des commerces.

La catégorie "Autre" montre une consommation stable, sans variations significatives au cours de la
journée. Cela pourrait inclure des usages résidentiels ou des infrastructures nécessitant un
approvisionnement constant.

Ce visuel met en évidence les différences de comportement entre les secteurs en termes de
consommation énergétique. Alors que la grande industrie nécessite une alimentation continue, la
consommation du secteur tertiaire est plus liée aux horaires de travail. Cette distinction est importante
pour optimiser la gestion de l'approvisionnement en électricité, notamment en ajustant la production
aux périodes de forte demande."""
    },
    "26. Relation entre la température moyenne et la consommation d'électricité en France": {
        "category": "Climat & Impact Énergie", # Ancienne catégorie
        "text": """Ce graphique de dispersion (scatter plot) illustre la relation entre la température moyenne (en °C) et
la consommation d'électricité (en MW) en France. On observe une relation non linéaire
caractéristique : la consommation d'électricité est plus élevée aux extrêmes de la courbe de
température, c'est-à-dire lorsque les températures sont très basses (en dessous de 5 °C) ou très
élevées (au-dessus de 20 °C). Cette relation traduit l'impact des besoins en chauffage et en
climatisation sur la consommation énergétique.

Lorsque les températures sont basses, la consommation d'électricité augmente de manière
significative, principalement en raison de l'utilisation accrue des systèmes de chauffage électrique.
Inversement, on observe également une augmentation de la consommation lorsque les températures
sont élevées, ce qui correspond à une demande accrue en climatisation et en ventilation.

La consommation d'électricité est plus modérée pour des températures comprises entre 10 °C et 20
°C, correspondant à une plage où les besoins de chauffage et de climatisation sont réduits.

Ce visuel met en lumière la forte dépendance de la consommation électrique aux conditions
climatiques. Il souligne l'importance de prévoir la gestion de la demande énergétique en fonction des
variations saisonnières de la température, notamment pour éviter des pics de consommation lors
d'épisodes de froid extrême ou de vagues de chaleur."""
    },
    "27. Consommation électrique moyenne par catégorie de température": {
        "category": "Climat & Impact Énergie", # Ancienne catégorie
        "text": """Ce graphique à barres présente la consommation électrique moyenne en fonction des catégories de
température. Les différentes catégories sont classées de "Très froid" (< 0 °C) à "Très chaud" (> 30
°C). On observe que la consommation électrique atteint son pic dans les conditions de "Très froid",
avec une consommation moyenne dépassant les 300 000 MW. Cette forte demande est due à
l'utilisation massive des systèmes de chauffage électrique pendant les périodes de températures très basses.

À l'inverse, les catégories de température "Modéré" (10-20 °C) et "Chaud" (20-30 °C) affichent les
consommations les plus faibles. Cela s'explique par le fait que dans cette plage de températures, les
besoins en chauffage et en climatisation sont réduits.

La consommation remonte légèrement dans la catégorie "Très chaud" (> 30 °C), en raison de
l'augmentation de l'utilisation des systèmes de climatisation pendant les vagues de chaleur.

Ce visuel met en évidence la corrélation entre les conditions climatiques extrêmes (froid ou chaud) et
la consommation énergétique. Les périodes de températures extrêmes entraînent une forte demande
en énergie, soulignant l'importance de prévoir des stratégies de gestion de la demande énergétique,
notamment en renforçant l'efficacité énergétique des bâtiments pour le chauffage et la climatisation."""
    },
    "28. Impact des variations de température sur la consommation électrique": {
        "category": "Climat & Impact Énergie", # Ancienne catégorie
        "text": """Ce graphique combine une courbe de température moyenne mensuelle (en rouge) et des barres
représentant la consommation électrique mensuelle (en bleu) en France. Il met en évidence la
relation inverse entre la température moyenne et la consommation électrique. En hiver, lorsque les
températures sont les plus basses (notamment en janvier et décembre), la consommation électrique
atteint son pic, principalement en raison des besoins accrus de chauffage.

À l'inverse, durant les mois les plus chauds (de juin à août), les températures atteignent leur pic, mais
la consommation électrique diminue légèrement. Cependant, on remarque que la consommation ne
baisse pas autant qu'on pourrait s'y attendre, ce qui peut être attribué à l'utilisation croissante des
climatiseurs pendant les vagues de chaleur estivales.

Ce visuel met en évidence l'importance des variations saisonnières sur la consommation électrique. Il
souligne la nécessité de gérer les pics de demande énergétique en hiver, tout en anticipant une
augmentation de la demande estivale liée au réchauffement climatique. Les politiques énergétiques
doivent prendre en compte ces variations saisonnières pour assurer un approvisionnement stable tout
au long de l'année."""
    }
}

# --- /!\ MODIFICATION : Suppression de "Analyses Régionales" ---
# --- Nouvelles Catégories Ordonnées (SANS Analyses Régionales) ---
NEW_CATEGORIES_ORDERED = [
    "📈 Consommation : Tendances & Rythmes",
    "🏭 Production & Mix Énergétique",
    # "🗺️ Analyses Régionales", # <-- SUPPRIMÉ
    "☀️ Climat & Météo : Impact Énergie",
    "🏢 Consommation par Secteur & Économie",
    "👨‍👩‍👧‍👦 Démographie"
]

# --- Mapping des anciennes catégories vers les nouvelles (pour la logique de regroupement) ---
# Note: 'Analyses Régionales (Énergie & Population)' n'a plus de destination directe.
# Sa logique sera gérée au cas par cas dans la boucle de regroupement ci-dessous.
OLD_TO_NEW_CATEGORY_MAP = {
    "Consommation (Tendances & Saisons)": "📈 Consommation : Tendances & Rythmes",
    "Production & Mix Énergétique": "🏭 Production & Mix Énergétique",
    "Analyses Régionales (Énergie & Population)": None, # Sera traité spécifiquement
    "Démographie Nationale": "👨‍👩‍👧‍👦 Démographie",
    "Climat & Impact Énergie": "☀️ Climat & Météo : Impact Énergie",
    "Consommation par Secteur": "🏢 Consommation par Secteur & Économie"
}

# --- Regrouper les visualisations par NOUVELLE catégorie ---
visualizations_by_new_category = {cat: [] for cat in NEW_CATEGORIES_ORDERED}
for original_key, details in visualizations_data.items():
    old_category = details["category"]
    new_category = None # Réinitialiser pour chaque visualisation

    # /!\ MODIFICATION : Logique de réaffectation des visualisations de l'ancienne catégorie "Analyses Régionales" ---
    if old_category == "Analyses Régionales (Énergie & Population)":
        # Réaffectation basée sur le contenu de la visualisation (clé originale)
        if original_key in ["3. Répartition régionale de la consommation totale d'énergie en France (2013 à 2023)",
                           "11. Répartition régionale de la consommation d'électricité"]:
             new_category = "📈 Consommation : Tendances & Rythmes"
        elif original_key in ["12. Production électrique : Disparités régionales"]:
             new_category = "🏭 Production & Mix Énergétique"
        elif original_key in ["13. Croissance démographique régionale (1990-2024)", # Pop 2024 map
                           "15. Répartition de la population par région en France au 31 décembre 2024",
                           "16. Croissance annuelle moyenne de la population par région (1990-2024)"]:
             new_category = "👨‍👩‍👧‍👦 Démographie"
        elif original_key in ["18. Carte des températures moyennes annuelles par région en France",
                           "19. Carte de chaleur des températures moyennes par région et mois en France"]:
             new_category = "☀️ Climat & Météo : Impact Énergie"
        elif original_key in ["23. Top 5 des régions avec le plus d'entreprises (2019)",
                           "24. Consommation d'électricité par secteur d'activité économique et par région (2023)"]:
             new_category = "🏢 Consommation par Secteur & Économie"
        else:
            # Fallback au cas où une visualisation de cette catégorie n'est pas listée ci-dessus
             print(f"Avertissement: Visualisation '{original_key}' de catégorie 'Analyses Régionales (Énergie & Population)' non explicitement réaffectée. Placement dans la première catégorie.")
             new_category = NEW_CATEGORIES_ORDERED[0] # Ou une autre catégorie par défaut
    else:
        # Pour les autres catégories, utiliser le mapping standard
        new_category = OLD_TO_NEW_CATEGORY_MAP.get(old_category)

    # Cas particulier: la 14 est Démographie Nationale
    if original_key == "14. Croissance de la population totale en France (1990-2024)":
        new_category = "👨‍👩‍👧‍👦 Démographie"


    if new_category and new_category in visualizations_by_new_category:
        visualizations_by_new_category[new_category].append(original_key)
    else:
        print(f"Avertissement: Impossible de mapper la visualisation '{original_key}' (ancienne cat: '{old_category}') vers une nouvelle catégorie valide ou catégorie non trouvée: '{new_category}'.")
        # Optionnel: Mettre dans une catégorie "Autre" ou la première par défaut si elle existe encore
        if NEW_CATEGORIES_ORDERED:
            visualizations_by_new_category[NEW_CATEGORIES_ORDERED[0]].append(original_key)


# Trier les visualisations dans chaque catégorie par leur numéro
for cat in visualizations_by_new_category:
    visualizations_by_new_category[cat].sort(key=lambda x: int(x.split('.')[0]))

# --- Fonction pour créer des titres d'affichage plus courts ---
def create_display_title(original_key):
    try:
        # Enlever le numéro et le point, garder le reste
        title_part = original_key.split('.', 1)[1].strip()
        # Raccourcissements simples (peut être amélioré)
        replacements = {
            "consommation d'énergie": "Conso. Énergie",
            "consommation d'électricité": "Conso. Élec.",
            "production d'électricité": "Prod. Élec.",
            "production d'énergie": "Prod. Énergie",
            "répartition régionale": "Répart. Régionale",
            "évolution temporelle": "Évol. Temporelle",
            "températures moyennes": "Temp. Moyennes",
            "secteur d'activité économique": "Secteur Éco.",
            "énergies renouvelables": "EnR",
            "France": "FR",
            "par région": "/ Région",
            "par saison": "/ Saison",
            "par mois": "/ Mois",
            "sur une journée": "/ Jour",
            "au niveau national": "(National)",
            "annuelles": "annuelles", # Garder ou raccourcir ?
            "mensuelle": "mensuelle"
        }
        for old, new in replacements.items():
            title_part = title_part.replace(old, new)

        # Limiter la longueur si nécessaire
        max_len = 70 # Ajuster si besoin
        if len(title_part) > max_len:
            title_part = title_part[:max_len-3] + "..."
        return title_part
    except Exception: # Fallback en cas d'erreur de split ou autre
        return original_key # Retourne la clé originale si le formatage échoue

# --- Créer un mapping Titre Affichage -> Clé Originale pour chaque catégorie ---
display_title_map_by_category = {}
for category, original_keys in visualizations_by_new_category.items():
    display_map = {"--- Choisir une visualisation ---": None} # Option par défaut
    for key in original_keys:
        display_map[create_display_title(key)] = key
    display_title_map_by_category[category] = display_map


# =============================================================================
# --- 7. MAIN APP LOGIC / PAGE CONTENT ---
# =============================================================================

current_choice = st.session_state.choix

# --- Introduction Section ---
if current_choice == "👋 Introduction":
    st.markdown("<h1 style='text-align: center; color: #5533FF;'>👋 Bienvenue sur notre projet de consommation d'électricité en France ⚡</h1>", unsafe_allow_html=True)

    st.markdown(
        """
        <p style='text-align: center; font-size: 18px;'>
        La consommation d'électricité est un enjeu majeur dans la transition énergétique. Ce projet explore les données françaises afin de mieux comprendre les tendances de consommation,
        visualiser les variations saisonnières, et prévoir la demande énergétique future. 🌱
        </p>
        """,
        unsafe_allow_html=True
    )

    st.write("""
    💡 **Pourquoi ce projet ?**  
    L'énergie occupe une place centrale dans les défis environnementaux et sociétaux contemporains. La France, avec son mix énergétique spécifique, offre une opportunité unique d'explorer
    les dynamiques de production et de consommation, particulièrement marquée par la prédominance du nucléaire et la croissance des énergies renouvelables.  
    Ce projet vise à répondre aux enjeux stratégiques suivants :
    - Étudier l'évolution temporelle de la consommation et production énergétiques en France.
    - Mettre en relation ces tendances avec des facteurs clés tels que la démographie, les conditions climatiques, et les activités économiques.
    - Développer des outils prédictifs pour mieux anticiper les besoins énergétiques et optimiser les ressources.
    """)

    st.write("""
    🔎 **Objectifs du projet :**
    - Explorer les données de consommation d'électricité en France
    - Créer des visualisations interactives des tendances énergétiques
    - Construire des modèles de machine learning basés sur les données historiques
    - Fournir des insights utiles pour la gestion de la demande en électricité
    """)

    if st.button("🚀 Commencer l'exploration"):
        next_section = "🔎 Exploration des données"
        st.session_state.choix = next_section
        st.rerun()

    # Affichage de l'image centrée avec encodage base64
        # <-- MODIFIÉ : Chemin relatif -->
    image_path = "Visualisation/sunrise-3579931_1280.jpg"
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()

        st.markdown(
            f"""
            <div style='display: flex; justify-content: center; margin-top: 30px;'>
                <img src="data:image/jpeg;base64,{encoded}" width="500"/>
            </div>
            <p style='text-align: center; font-size: 14px; color: gray;'>Énergie renouvelable au lever du soleil 🌄</p>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("❗ L’image n’a pas été trouvée à l’emplacement spécifié :")
        st.code(image_path)

# --- Data Exploration Section ---
elif current_choice == "🔎 Exploration des données":
    st.markdown("<h1 style='text-align: left;'>🔎 Exploration des Données</h1>", unsafe_allow_html=True)
    st.subheader("Sélectionnez un dataset à explorer")

    # --- Dictionnaire associant nom convivial et fonction getter ---
    dataset_options = {
        "📁 Consommation/Production d'éléctricité en France": get_eco2mix_data,
        "🏢 Nombre d'Entreprises/Établissements": get_effectifs_data,
        "🌡️ Température Quotidienne Régionale": get_temperature_data,
        "👪 Population Régionale": get_population_data, # Nom clarifié
        "⚡ Soutirages éléctrique régionaux quotidiens": get_soutirage_data
    }
    dataset_names = list(dataset_options.keys())

    selected_dataset_name = st.selectbox(
        "Choisissez un dataset :",
        dataset_names,
        key="dataset_selector"
        # index=0 # Optionnel: pour sélectionner le premier par défaut
    )

    df_to_display = None
    if selected_dataset_name:
        getter_function = dataset_options[selected_dataset_name]
        # L'appel st.spinner est correct ici, car il est en dehors de la fonction cachée
        with st.spinner(f"Chargement et traitement des données '{selected_dataset_name}'..."):
            start_time = time.time()
            # Appel de la fonction getter qui gère le cache et le chargement interne
            df_to_display = getter_function() # Appelle _load_eco2mix_internal (nettoyée) si cache vide
            end_time = time.time()
            print(f"--- Time taken for getter '{selected_dataset_name}': {end_time - start_time:.4f} seconds ---")

    # --- Affichage conditionnel basé sur le dataset sélectionné ---
    # Ces appels st.warning/st.error sont corrects ici, car ils sont en dehors de la fonction cachée
    if df_to_display is not None:
        if df_to_display.empty:
             st.warning(f"Le dataset '{selected_dataset_name}' est vide ou n'a pas pu être chargé/traité correctement.")
             st.info("Vérifiez si le fichier source existe, n'est pas vide et si les étapes de prétraitement n'ont pas supprimé toutes les lignes.")
        else:
            # --- Affichage spécifique pour Eco2mix ---
            # (Code inchangé ici, car les st.* sont déjà en dehors de la fonction cachée)
            if selected_dataset_name == "📁 Consommation/Production d'éléctricité en France":
                st.markdown("---")
                st.markdown(f"### 📝 Aperçu: {selected_dataset_name}")
                st.write("Ce dataset contient des données (souvent horaires ou demi-horaires) sur la consommation, la production par filière, et les échanges inter-régionaux/internationaux.")
                st.write(f"Nombre total de lignes: {len(df_to_display):,}".replace(",", " "))
                st.write("Affichage des 5 premières lignes :")
                st.dataframe(df_to_display.head())

                st.markdown("---")
                st.markdown("### 📊 Informations Générales")
                st.write("Types de données des colonnes :")
                st.dataframe(df_to_display.dtypes.astype(str).reset_index().rename(columns={'index': 'Colonne', 0: 'Type'}))
                st.write("Statistiques descriptives (colonnes numériques) :")
                try:
                    st.dataframe(df_to_display.describe())
                except Exception as e:
                    st.warning(f"Impossible d'afficher les statistiques descriptives: {e}")


                st.markdown("---")
                st.markdown("### ❓ Analyse des valeurs manquantes")
                missing_values = df_to_display.isnull().sum()
                missing_df = missing_values[missing_values > 0].sort_values(ascending=False).reset_index()
                missing_df.columns = ['Colonne', 'Nombre Manquant']
                if not missing_df.empty:
                    st.write("Nombre de valeurs manquantes par colonne (colonnes avec > 0 manquants) :")
                    st.dataframe(missing_df)
                    total_missing = missing_df['Nombre Manquant'].sum()
                    total_cells = np.prod(df_to_display.shape)
                    percent_missing = (total_missing / total_cells) * 100 if total_cells > 0 else 0
                    st.write(f"Total des valeurs manquantes : {total_missing:,} ({percent_missing:.2f}% du total des cellules)".replace(",", " "))
                else:
                    st.success("✅ Aucune valeur manquante détectée dans ce dataset.")

                st.markdown("---")
                st.markdown("### 🗓️ Consulter la consommation par région et date")
                # Vérifier que les colonnes nécessaires existent ET ont le bon type
                required_cols = ['Région', 'Date', 'Consommation (MW)']
                if all(col in df_to_display.columns for col in required_cols):
                     if pd.api.types.is_datetime64_any_dtype(df_to_display['Date']):
                         # S'assurer que Consommation est numérique (peut avoir été lue comme object)
                         if not pd.api.types.is_numeric_dtype(df_to_display['Consommation (MW)']):
                             df_to_display['Consommation (MW)'] = pd.to_numeric(df_to_display['Consommation (MW)'], errors='coerce')

                         regions = sorted(df_to_display['Région'].dropna().unique())
                         if regions:
                             selected_region = st.selectbox("Sélectionner une Région:", regions, key="eco2mix_region_select")
                             # Utiliser les vraies min/max dates du DF pour le date_input
                             min_date = df_to_display['Date'].min().date()
                             max_date = df_to_display['Date'].max().date()
                             # Proposer la date la plus récente par défaut
                             default_date = max_date if max_date >= min_date else min_date

                             selected_date_input = st.date_input(
                                 "Sélectionner une Date:",
                                 min_value=min_date,
                                 max_value=max_date,
                                 value=default_date,
                                 key="eco2mix_date_select"
                             )
                             # Convertir la date sélectionnée en objet date pour la comparaison
                             selected_date = pd.to_datetime(selected_date_input).date()

                             # Filtrer le DataFrame
                             filtered_df_day = df_to_display[
                                 (df_to_display['Région'] == selected_region) &
                                 (df_to_display['Date'].dt.date == selected_date)
                             ].copy() # Utiliser .copy() pour éviter SettingWithCopyWarning

                             if filtered_df_day.empty:
                                 st.info(f"Aucune donnée de consommation trouvée pour '{selected_region}' le {selected_date_input.strftime('%d/%m/%Y')}.")
                                 total_consumption = 0
                             else:
                                 # Recalculer la somme sur le df filtré (la colonne devrait déjà être numérique)
                                 total_consumption = filtered_df_day['Consommation (MW)'].sum(skipna=True)

                             # Formater pour l'affichage
                             try:
                                 formatted_consumption = "{:,.0f}".format(total_consumption).replace(",", " ") if pd.notna(total_consumption) else "N/A"
                             except (ValueError, TypeError):
                                 formatted_consumption = "Erreur Format"

                             # Affichage amélioré
                             st.markdown(f"""
                                        <div style='background-color: #262730; padding: 20px; border-radius: 10px; border: 1px solid #333; text-align: center; margin-top: 15px;'>
                                            <h4 style='font-weight: 500; color: #FAFAFA; margin-bottom: 12px;'>💡 Consommation Totale (Journalière)</h4>
                                            <p style='color: #AAA; font-size: 16px; margin-bottom: 8px;'>Pour <span style='color: #A0DAFF; font-weight: bold;'>{selected_region}</span> le <span style='color: #A0DAFF; font-weight: bold;'>{selected_date_input.strftime('%d/%m/%Y')}</span></p>
                                            <h2 style='color: #FFD700; font-weight: bold; letter-spacing: 1px;'>{formatted_consumption} MW</h2>
                                            <p style='font-size: 0.8em; color: #777; margin-top: 10px;'>(Somme des relevés disponibles pour ce jour)</p>
                                        </div>""", unsafe_allow_html=True)
                         else:
                            st.warning("Aucune région unique trouvée dans les données pour le filtrage.")
                     else:
                        st.error("La colonne 'Date' n'est pas au format datetime après chargement. Vérifiez le fichier source ou la fonction de chargement.")
                else:
                    st.error(f"Colonnes requises ({', '.join(required_cols)}) manquantes pour cette analyse.")

            # --- Affichage spécifique pour Effectifs ---
            elif selected_dataset_name == "🏢 Nombre d'Entreprises/Établissements":
                # (Code inchangé ici)
                st.markdown("---")
                st.markdown("### 📜 Aperçu: Base Établissement par Tranche d'Effectif")
                st.write("""
                    Ce dataset recense le nombre d'établissements par commune (identifiée par `CODGEO`, `LIBGEO`), ventilé selon différentes tranches d'effectifs salariés.
                    Il donne une image de la structure économique locale. Il inclut aussi les codes Région (`REG`) et Département (`DEP`).
                    Colonnes `E14TST`: Total, `E14TS0ND`: 0 salariés, `E14TS1`: 1-5, `E14TS6`: 6-9, etc.
                """)
                st.write(f"Nombre total de lignes (communes/entités): {len(df_to_display):,}".replace(",", " "))
                st.write("Affichage des 5 premières lignes :")
                st.dataframe(df_to_display.head())

                st.markdown("---")
                st.markdown("### 📊 Informations Générales")
                st.write("Types de données des colonnes :")
                st.dataframe(df_to_display.dtypes.astype(str).reset_index().rename(columns={'index': 'Colonne', 0: 'Type'}))
                st.write("Statistiques descriptives (colonnes numériques - nombre d'établissements) :")
                try:
                    num_cols_effectifs_present = [col for col in ['E14TST', 'E14TS0ND', 'E14TS1', 'E14TS6', 'E14TS10', 'E14TS20', 'E14TS50', 'E14TS100', 'E14TS200', 'E14TS500'] if col in df_to_display.columns]
                    if num_cols_effectifs_present:
                        st.dataframe(df_to_display[num_cols_effectifs_present].describe())
                    else:
                        st.warning("Aucune colonne d'effectif numérique standard trouvée pour les statistiques.")
                except Exception as e:
                    st.warning(f"Impossible d'afficher les statistiques descriptives: {e}")


                st.markdown("---")
                st.markdown("### ❓ Analyse des valeurs manquantes")
                missing_values = df_to_display.isnull().sum()
                missing_df = missing_values[missing_values > 0].sort_values(ascending=False).reset_index()
                missing_df.columns = ['Colonne', 'Nombre Manquant']
                if not missing_df.empty:
                    st.write("Nombre de valeurs manquantes par colonne (après conversion numérique) :")
                    st.dataframe(missing_df)
                else:
                    st.success("✅ Aucune valeur manquante détectée dans ce dataset après traitement initial.")

                st.markdown("---")
                st.markdown("### 📍 Consulter les effectifs agrégés par Département")
                effectifs_req_cols = ['DEP', 'LIBGEO'] # Minimal pour le filtre/libellé
                num_cols_effectifs = ['E14TST', 'E14TS0ND', 'E14TS1', 'E14TS6', 'E14TS10', 'E14TS20', 'E14TS50', 'E14TS100', 'E14TS200', 'E14TS500']
                # Vérifier que DEP et LIBGEO existent
                if all(col in df_to_display.columns for col in effectifs_req_cols):
                     # S'assurer que DEP est bien de type string pour le filtrage et l'affichage correct
                     if not pd.api.types.is_string_dtype(df_to_display['DEP']):
                         st.warning("La colonne 'DEP' n'est pas de type string. Tentative de correction pour le filtre...")
                         # Appliquer la même logique de correction que dans le loader au cas où
                         try:
                             df_to_display['DEP'] = df_to_display['DEP'].fillna('NA').astype(str).str.replace(r'\.0$', '', regex=True)
                             df_to_display['DEP'] = df_to_display['DEP'].apply(lambda x: x.zfill(2) if x != 'NA' and x.isdigit() else x)
                             df_to_display['DEP'] = df_to_display['DEP'].replace('NA', pd.NA) # Remettre NA si c'était NaN
                         except Exception as e_conv_filter:
                             st.error(f"Erreur lors de la tentative de correction de la colonne DEP: {e_conv_filter}. Le filtre risque de ne pas fonctionner.")
                             st.stop() # Arrêter si la colonne clé n'est pas utilisable

                     # Obtenir la liste unique des départements valides (ignorer les NA)
                     deps = sorted(df_to_display['DEP'].dropna().unique())

                     if deps:
                         # Créer un mapping pour afficher "Code - Nom Exemple" dans le selectbox
                         # Trouver une commune représentative (ex: la plus grande E14TST) pour chaque DEP
                         dep_labels = {}
                         if 'E14TST' in df_to_display.columns and pd.api.types.is_numeric_dtype(df_to_display['E14TST']):
                              try:
                                # idx = df_to_display.groupby('DEP')['E14TST'].idxmax() # Peut échouer si NaN ou empty groups
                                # representative_communes = df_to_display.loc[idx][['DEP', 'LIBGEO']].set_index('DEP')['LIBGEO']
                                # Méthode plus sûre: itérer sur les deps uniques
                                for d in deps:
                                     communes_in_dep = df_to_display[df_to_display['DEP'] == d]
                                     if not communes_in_dep.empty:
                                         # Prendre la première commune par ordre alphabétique comme exemple
                                         # Ou celle avec le plus d'établissements si E14TST est fiable
                                         # best_commune = communes_in_dep.loc[communes_in_dep['E14TST'].idxmax()]['LIBGEO'] if not communes_in_dep['E14TST'].isnull().all() else communes_in_dep.iloc[0]['LIBGEO']
                                         best_commune = communes_in_dep.sort_values(by='LIBGEO').iloc[0]['LIBGEO']
                                         dep_labels[d] = f"{d} - (Ex: {best_commune})"
                                     else:
                                         dep_labels[d] = f"{d} - (N/A)"

                              except Exception as e_repr:
                                  print(f"Warning: Could not generate representative commune names: {e_repr}")
                                  # Fallback: juste afficher le code
                                  dep_labels = {d: d for d in deps}
                         else:
                             # Fallback si E14TST n'existe pas ou n'est pas numérique
                             print("Warning: Column E14TST not suitable for finding representative communes. Using first commune found.")
                             for d in deps:
                                 first_commune = df_to_display[df_to_display['DEP'] == d].iloc[0]['LIBGEO'] if not df_to_display[df_to_display['DEP'] == d].empty else "N/A"
                                 dep_labels[d] = f"{d} - (Ex: {first_commune})"


                         selected_dep_code = st.selectbox(
                             "Sélectionner un Département:",
                             options=deps,
                             format_func=lambda x: dep_labels.get(x, x), # Afficher "Code - Ex: Nom"
                             key="effectifs_dep_select",
                             index=0 # Sélectionner le premier par défaut
                         )

                         if selected_dep_code:
                             # Filtrer les données pour le département choisi
                             filtered_dep_df = df_to_display[df_to_display['DEP'] == selected_dep_code].copy()

                             if not filtered_dep_df.empty:
                                 # Calculer les totaux pour les colonnes numériques existantes dans ce département
                                 sums = {}
                                 for col in num_cols_effectifs:
                                     if col in filtered_dep_df.columns and pd.api.types.is_numeric_dtype(filtered_dep_df[col]):
                                         # Assurer la somme sur des numériques, ignorer NaN
                                         sums[col] = filtered_dep_df[col].sum(skipna=True)
                                     else:
                                         sums[col] = 0 # Mettre 0 si la colonne manque ou n'est pas numérique

                                 # Afficher les métriques résumées
                                 st.markdown(f"""
                                     <div style='background-color: #262730; padding: 20px; border-radius: 10px; border: 1px solid #333; margin-top: 15px;'>
                                         <h4 style='font-weight: 500; color: #FAFAFA; text-align: center; margin-bottom: 15px;'>🏢 Structure des Établissements (Agrégée)</h4>
                                         <p style='color: #AAA; font-size: 16px; text-align: center; margin-bottom: 15px;'>
                                             Pour le département <span style='color: #A0DAFF; font-weight: bold;'>{dep_labels.get(selected_dep_code, selected_dep_code)}</span>
                                         </p>
                                         <div style='display: flex; justify-content: space-around; text-align: center; flex-wrap: wrap;'>
                                             <div style='margin: 5px 10px;'>
                                                 <p style='color: #FFF; font-size:0.9em; margin-bottom: 5px;'>Total Étab.</p>
                                                 <h3 style='color: #FFD700; margin-top: 0;'>{sums.get('E14TST', 0):,.0f}</h3>
                                             </div>
                                             <div style='margin: 5px 10px;'>
                                                 <p style='color: #AAA; font-size:0.9em; margin-bottom: 5px;'>0 salariés</p>
                                                 <h3 style='color: #ADD8E6; margin-top: 0;'>{sums.get('E14TS0ND', 0):,.0f}</h3>
                                             </div>
                                             <div style='margin: 5px 10px;'>
                                                 <p style='color: #AAA; font-size:0.9em; margin-bottom: 5px;'>1-5</p>
                                                 <h3 style='color: #90EE90; margin-top: 0;'>{sums.get('E14TS1', 0):,.0f}</h3>
                                             </div>
                                             <div style='margin: 5px 10px;'>
                                                 <p style='color: #AAA; font-size:0.9em; margin-bottom: 5px;'>6-9</p>
                                                 <h3 style='color: #FFB6C1; margin-top: 0;'>{sums.get('E14TS6', 0):,.0f}</h3>
                                             </div>
                                             <div style='margin: 5px 10px;'>
                                                 <p style='color: #AAA; font-size:0.9em; margin-bottom: 5px;'>10-19</p>
                                                 <h3 style='color: #FFA07A; margin-top: 0;'>{sums.get('E14TS10', 0):,.0f}</h3>
                                             </div>
                                              <div style='margin: 5px 10px;'>
                                                 <p style='color: #AAA; font-size:0.9em; margin-bottom: 5px;'>20-49</p>
                                                 <h3 style='color: #B19CD9; margin-top: 0;'>{sums.get('E14TS20', 0):,.0f}</h3>
                                             </div>
                                             <div style='margin: 5px 10px;'>
                                                 <p style='color: #AAA; font-size:0.9em; margin-bottom: 5px;'>50-99</p>
                                                 <h3 style='color: #FFDB58; margin-top: 0;'>{sums.get('E14TS50', 0):,.0f}</h3>
                                             </div>
                                             <div style='margin: 5px 10px;'>
                                                 <p style='color: #AAA; font-size:0.9em; margin-bottom: 5px;'>100-199</p>
                                                 <h3 style='color: #F0E68C; margin-top: 0;'>{sums.get('E14TS100', 0):,.0f}</h3>
                                             </div>
                                             # Ajouter E14TS200 et E14TS500 si pertinent
                                         </div>
                                     </div>
                                 """, unsafe_allow_html=True)

                                 # Ajouter un graphique à barres pour la distribution des tranches
                                 st.markdown("---")
                                 st.write("Répartition par tranche d'effectif (Nombre total d'établissements dans le département) :")
                                 tranche_labels_map = {
                                     '0 salariés': 'E14TS0ND', '1-5': 'E14TS1', '6-9': 'E14TS6',
                                     '10-19': 'E14TS10', '20-49': 'E14TS20', '50-99': 'E14TS50',
                                     '100-199': 'E14TS100', '200-499': 'E14TS200', '500+': 'E14TS500'
                                 }
                                 # Utiliser les sommes déjà calculées (sums) en filtrant sur les colonnes existantes
                                 tranche_sums_chart = {}
                                 for label, col_name in tranche_labels_map.items():
                                     if col_name in sums: # Utilise les sommes déjà calculées et validées
                                         tranche_sums_chart[label] = sums[col_name]

                                 if tranche_sums_chart:
                                     # Créer une Series pour le graphique
                                     tranche_sums_series = pd.Series(tranche_sums_chart)
                                     st.bar_chart(tranche_sums_series)
                                 else:
                                     st.warning("Aucune donnée de tranche d'effectif valide à afficher pour ce département.")

                             else:
                                 st.info(f"Aucune donnée d'effectif trouvée pour le département {selected_dep_code}.")
                     else:
                         st.warning("Aucun code département valide trouvé dans les données pour permettre le filtrage.")
                else:
                     st.error(f"Colonnes requises ('DEP', 'LIBGEO') manquantes pour le filtre des effectifs par département.")


            # --- Affichage spécifique pour Soutirage ---
            elif selected_dataset_name == "⚡ Soutirages éléctrique régionaux quotidiens":
                # (Code inchangé ici)
                st.markdown("---")
                st.markdown(f"### 📝 Aperçu: {selected_dataset_name}")
                st.write("Ce dataset fournit les volumes d'énergie électrique 'soutirés' (consommés) du réseau, agrégés par jour et par région.")
                st.write(f"Nombre total de lignes: {len(df_to_display):,}".replace(",", " "))
                st.write("Affichage des 5 premières lignes :")
                st.dataframe(df_to_display.head())

                st.markdown("---")
                st.markdown("### 📊 Informations Générales")
                st.write("Types de données des colonnes :")
                st.dataframe(df_to_display.dtypes.astype(str).reset_index().rename(columns={'index': 'Colonne', 0: 'Type'}))
                st.write("Statistiques descriptives (colonnes numériques) :")
                try:
                    st.dataframe(df_to_display.describe())
                except Exception as e:
                    st.warning(f"Impossible d'afficher les statistiques descriptives: {e}")


                st.markdown("---")
                st.markdown("### ❓ Analyse des valeurs manquantes")
                missing_values = df_to_display.isnull().sum()
                missing_df = missing_values[missing_values > 0].sort_values(ascending=False).reset_index()
                missing_df.columns = ['Colonne', 'Nombre Manquant']
                if not missing_df.empty:
                     st.write("Nombre de valeurs manquantes par colonne:")
                     st.dataframe(missing_df)
                else:
                     st.success("✅ Aucune valeur manquante détectée.")
                # Ajouter ici une section de filtrage/visualisation si pertinent (similaire à Eco2mix/Temp)


            # --- Affichage spécifique pour Température ---
            elif selected_dataset_name == "🌡️ Température Quotidienne Régionale":
                # (Code inchangé ici)
                st.markdown("---")
                st.markdown("### 📜 Aperçu: Température Quotidienne Régionale")
                st.write("""
                    Ce dataset contient des relevés météorologiques quotidiens par région (TMin, TMax, TMoy).
                    Essentiel pour corréler la météo et la consommation d'énergie.
                """)
                st.write(f"Nombre total de lignes: {len(df_to_display):,}".replace(",", " "))
                st.write("Affichage des 5 premières lignes :")
                cols_to_show_temp = ['Date', 'Région', 'TMin (°C)', 'TMax (°C)', 'TMoy (°C)']
                st.dataframe(df_to_display[[col for col in cols_to_show_temp if col in df_to_display.columns]].head())

                st.markdown("---")
                st.markdown("### 📊 Informations Générales")
                st.write("Types de données des colonnes :")
                st.dataframe(df_to_display.dtypes.astype(str).reset_index().rename(columns={'index': 'Colonne', 0: 'Type'}))
                st.write("Statistiques descriptives (colonnes numériques) :")
                try:
                    temp_num_cols = ['TMin (°C)', 'TMax (°C)', 'TMoy (°C)']
                    st.dataframe(df_to_display[[col for col in temp_num_cols if col in df_to_display.columns]].describe())
                except Exception as e:
                    st.warning(f"Impossible d'afficher les statistiques descriptives: {e}")


                st.markdown("---")
                st.markdown("### ❓ Analyse des valeurs manquantes")
                missing_values = df_to_display.isnull().sum()
                missing_df = missing_values[missing_values > 0].sort_values(ascending=False).reset_index()
                missing_df.columns = ['Colonne', 'Nombre Manquant']
                if not missing_df.empty:
                    st.write("Nombre de valeurs manquantes par colonne:")
                    st.dataframe(missing_df)
                else:
                    st.success("✅ Aucune valeur manquante détectée.")

                st.markdown("---")
                st.markdown("### 🌡️ Consulter la température par région et date")
                temp_req_cols = ['Région', 'Date', 'TMin (°C)', 'TMax (°C)', 'TMoy (°C)']
                if all(col in df_to_display.columns for col in temp_req_cols):
                    if pd.api.types.is_datetime64_any_dtype(df_to_display['Date']):
                         # Vérifier que les colonnes de température sont numériques
                         for col in ['TMin (°C)', 'TMax (°C)', 'TMoy (°C)']:
                             if not pd.api.types.is_numeric_dtype(df_to_display[col]):
                                 df_to_display[col] = pd.to_numeric(df_to_display[col], errors='coerce')

                         regions_temp = sorted(df_to_display['Région'].dropna().unique())
                         if regions_temp:
                            selected_region_temp = st.selectbox("Sélectionner une Région:", regions_temp, key="temp_region_select")
                            min_date_temp = df_to_display['Date'].min().date()
                            max_date_temp = df_to_display['Date'].max().date()
                            default_date_temp = max_date_temp if max_date_temp >= min_date_temp else min_date_temp
                            selected_date_temp_input = st.date_input(
                                "Sélectionner une Date:",
                                min_value=min_date_temp,
                                max_value=max_date_temp,
                                value=default_date_temp,
                                key="temp_date_select"
                            )
                            selected_date_temp = pd.to_datetime(selected_date_temp_input).date()

                            # Filtrer les données
                            filtered_temp_day = df_to_display[
                                (df_to_display['Région'] == selected_region_temp) &
                                (df_to_display['Date'].dt.date == selected_date_temp)
                            ]

                            if not filtered_temp_day.empty:
                                # Il peut y avoir plusieurs enregistrements pour une même date/région? Prendre le premier.
                                temp_data = filtered_temp_day.iloc[0]
                                # Utiliser .get() pour la robustesse si une colonne manque malgré le check initial
                                tmin = temp_data.get('TMin (°C)', np.nan)
                                tmax = temp_data.get('TMax (°C)', np.nan)
                                tmoy = temp_data.get('TMoy (°C)', np.nan)
                                # Formater pour affichage, gérer les NaN
                                tmin_str = f"{tmin:.1f}°C" if pd.notna(tmin) else "N/A"
                                tmax_str = f"{tmax:.1f}°C" if pd.notna(tmax) else "N/A"
                                tmoy_str = f"{tmoy:.1f}°C" if pd.notna(tmoy) else "N/A"

                                st.markdown(f"""
                                    <div style='background-color: #262730; padding: 20px; border-radius: 10px; border: 1px solid #333; margin-top: 15px;'>
                                        <h4 style='font-weight: 500; color: #FAFAFA; text-align: center; margin-bottom: 15px;'>🌡️ Températures Enregistrées</h4>
                                        <p style='color: #AAA; font-size: 16px; text-align: center; margin-bottom: 15px;'>Pour <span style='color: #A0DAFF; font-weight: bold;'>{selected_region_temp}</span> le <span style='color: #A0DAFF; font-weight: bold;'>{selected_date_temp_input.strftime('%d/%m/%Y')}</span></p>
                                        <div style='display: flex; justify-content: space-around; text-align: center; flex-wrap: wrap;'>
                                            <div style='margin: 5px 10px;'><p style='color: #ADD8E6; margin-bottom: 5px; font-size:0.9em;'>Minimale</p><h3 style='color: #87CEEB; margin-top: 0;'>{tmin_str}</h3></div>
                                            <div style='margin: 5px 10px;'><p style='color: #FFD700; margin-bottom: 5px; font-size:0.9em;'>Moyenne</p><h3 style='color: #FFFFE0; margin-top: 0;'>{tmoy_str}</h3></div>
                                            <div style='margin: 5px 10px;'><p style='color: #FFB6C1; margin-bottom: 5px; font-size:0.9em;'>Maximale</p><h3 style='color: #FFA07A; margin-top: 0;'>{tmax_str}</h3></div>
                                        </div>
                                    </div>""", unsafe_allow_html=True)
                            else:
                                st.info(f"Aucune donnée de température trouvée pour '{selected_region_temp}' le {selected_date_temp_input.strftime('%d/%m/%Y')}.")
                         else:
                            st.warning("Aucune région unique trouvée pour le filtrage température.")
                    else:
                        st.error("La colonne 'Date' n'est pas au format datetime après chargement (Température).")
                else:
                    st.error(f"Colonnes requises ({', '.join(temp_req_cols)}) manquantes pour le filtre température.")


            # --- Affichage spécifique pour Population ---
            elif selected_dataset_name == "👪 Population Régionale":
                # (Code inchangé ici)
                st.markdown("---")
                st.markdown(f"### 📝 Aperçu: {selected_dataset_name}")
                st.write("Données de population par région (source INSEE), transformées au format 'long' (une ligne par région par date/année).")
                st.write(f"Nombre total de lignes: {len(df_to_display):,}".replace(",", " "))
                st.write("Affichage des 5 premières lignes (après transformation) :")
                st.dataframe(df_to_display.head())

                st.markdown("---")
                st.markdown("### 📊 Informations Générales")
                st.write("Types de données des colonnes :")
                st.dataframe(df_to_display.dtypes.astype(str).reset_index().rename(columns={'index': 'Colonne', 0: 'Type'}))
                st.write("Statistiques descriptives (Population) :")
                try:
                    if 'Population' in df_to_display.columns:
                         st.dataframe(df_to_display[['Population']].describe())
                    else:
                         st.warning("Colonne 'Population' non trouvée pour les statistiques.")
                except Exception as e:
                    st.warning(f"Impossible d'afficher les statistiques descriptives: {e}")

                st.markdown("---")
                st.markdown("### ❓ Analyse des valeurs manquantes (après transformation)")
                missing_values = df_to_display.isnull().sum()
                missing_df = missing_values[missing_values > 0].sort_values(ascending=False).reset_index()
                missing_df.columns = ['Colonne', 'Nombre Manquant']
                if not missing_df.empty:
                     st.write("Nombre de valeurs manquantes par colonne :")
                     st.dataframe(missing_df)
                else:
                     st.success("✅ Aucune valeur manquante détectée après traitement.")

                st.markdown("---")
                st.markdown("### 👪 Consulter la population par région et année")
                pop_req_cols = ['Région', 'Date', 'Population']
                if all(col in df_to_display.columns for col in pop_req_cols):
                    if pd.api.types.is_datetime64_any_dtype(df_to_display['Date']):
                        if not pd.api.types.is_numeric_dtype(df_to_display['Population']):
                            df_to_display['Population'] = pd.to_numeric(df_to_display['Population'], errors='coerce')

                        regions_pop = sorted(df_to_display['Région'].dropna().unique())
                        # Les données de pop sont souvent annuelles, extraire les années uniques
                        years_pop = sorted(df_to_display['Date'].dt.year.dropna().unique())

                        if regions_pop and years_pop:
                            selected_region_pop = st.selectbox("Sélectionner une Région:", regions_pop, key="pop_region_select")
                            selected_year_pop = st.selectbox("Sélectionner une Année:", years_pop, key="pop_year_select")

                            # Filtrer par région ET année
                            filtered_pop = df_to_display[
                                (df_to_display['Région'] == selected_region_pop) &
                                (df_to_display['Date'].dt.year == selected_year_pop)
                            ].copy()

                            population_value = np.nan
                            date_display = f"l'année {selected_year_pop}" # Default display

                            if not filtered_pop.empty:
                                # S'il y a plusieurs entrées pour l'année (peu probable mais possible), prendre la première/dernière ? Prendre la première.
                                population_value = filtered_pop['Population'].iloc[0]
                                # Tenter d'afficher la date exacte si disponible
                                date_exact = filtered_pop['Date'].iloc[0]
                                if pd.notna(date_exact):
                                    date_display = date_exact.strftime('%d/%m/%Y')
                            else:
                                population_value = 0 # Ou laisser NaN/afficher "Non trouvée"

                            # Formater la population
                            if pd.notna(population_value):
                                try:
                                    formatted_population = "{:,.0f}".format(population_value).replace(",", " ")
                                except (ValueError, TypeError): formatted_population = "Erreur Format"
                            else:
                                formatted_population = "Donnée non trouvée"


                            st.markdown(f"""
                                <div style='background-color: #262730; padding: 20px; border-radius: 10px; border: 1px solid #333; text-align: center; margin-top: 15px;'>
                                    <h4 style='font-weight: 500; color: #FAFAFA; margin-bottom: 12px;'>👥 Population Estimée (Source: INSEE)</h4>
                                     <p style='color: #AAA; font-size: 16px; margin-bottom: 8px;'>Pour <span style='color: #A0DAFF; font-weight: bold;'>{selected_region_pop}</span> pour <span style='color: #A0DAFF; font-weight: bold;'>{date_display}</span></p>
                                    <h2 style='color: #90EE90; font-weight: bold; letter-spacing: 1px;'>{formatted_population} habitants</h2>
                                </div>""", unsafe_allow_html=True)
                        else:
                            st.warning("Impossible d'extraire les régions ou les années uniques pour le filtre population.")
                    else:
                        st.error("La colonne 'Date' n'est pas au format datetime après chargement (Population).")
                else:
                    st.error(f"Colonnes requises ({', '.join(pop_req_cols)}) manquantes pour cette analyse (Population).")

    elif selected_dataset_name:
        # Ce cas est atteint si df_to_display est None (erreur de chargement retournée par le getter)
        st.error(f"Le chargement ou le traitement initial du dataset '{selected_dataset_name}' a échoué.")
        st.info("Veuillez vérifier le chemin d'accès au fichier, son format (CSV avec séparateur attendu , ou ;), son encodage (UTF-8 ou Latin-1), et consultez les messages d'erreur détaillés dans la console pour plus d'indices.")
        # Afficher l'erreur traceback si disponible (utile pour le débogage)
        # Note: En production, on pourrait vouloir masquer traceback.
        # Vérifier si une exception a été capturée (vous pourriez la stocker dans session_state si nécessaire)
        # Si traceback n'est pas directement disponible ici, le message d'erreur de la console est la meilleure piste.
        # st.code(traceback.format_exc()) # Cette ligne ne fonctionnera que si l'exception est levée ici.


# --- Data Visualisation Section ---
elif current_choice == "📊 Data Visualisation":
    section_display_name = SECTION_ICONS.get(current_choice, current_choice)
    st.markdown(f"<h1 style='text-align: left;'>{section_display_name}</h1>", unsafe_allow_html=True)
    st.write("Explorez les différentes visualisations organisées par thème.")
    st.markdown("---")

    # /!\ MODIFICATION : Utilise la liste NEW_CATEGORIES_ORDERED mise à jour (sans Analyses Régionales)
    # Créer les onglets pour les nouvelles catégories
    tabs = st.tabs(NEW_CATEGORIES_ORDERED)

    # Parcourir chaque catégorie (onglet) et afficher le contenu correspondant
    for i, category_name in enumerate(NEW_CATEGORIES_ORDERED):
        with tabs[i]:
            st.subheader(f"Visualisations : {category_name}")

            # Récupérer le mapping titre affichage -> clé originale pour cette catégorie
            # Ce mapping a été mis à jour par la logique de regroupement précédente
            current_display_map = display_title_map_by_category.get(category_name, {})
            display_titles_options = list(current_display_map.keys())

            if len(display_titles_options) > 1: # Vérifier s'il y a des visualisations (plus que juste "--- Choisir ---")
                # Menu déroulant pour sélectionner une visualisation dans cet onglet
                selected_display_title = st.selectbox(
                    f"Choisissez une visualisation pour '{category_name}':",
                    options=display_titles_options,
                    key=f"visu_select_{i}", # Clé unique pour chaque selectbox
                    index=0 # Sélectionner "--- Choisir ---" par défaut
                )

                # Récupérer la clé originale correspondante
                original_key = current_display_map.get(selected_display_title)

                # --- Affichage de l'image et du texte si une visualisation est choisie ---
                if original_key:
                    try:
                        visu_number = original_key.split('.')[0]
                        image_filename = f"{visu_number}.png"
                        # Adapter le chemin si vos images sont ailleurs
                        image_path = os.path.join('Visualisation', image_filename)

                        st.markdown("---") # Séparateur avant la visualisation

                        # Colonnes pour l'image et le texte (optionnel, pour la mise en page)
                        col1, col2 = st.columns([1, 1]) # Version (Image 1/2, Texte 1/2)

                        with col1:
                            if os.path.exists(image_path):
                                st.image(image_path, caption=f"Visualisation : {original_key}")
                            else:
                                st.warning(f"Image non trouvée : {image_path}")
                                st.info(f"Assurez-vous que l'image '{image_filename}' existe dans le dossier 'Visualisation'.")

                        with col2:
                            description = visualizations_data[original_key]["text"]
                            st.markdown(f"#### Analyse")
                            st.write(description) # Le texte s'étalera maintenant dans une colonne plus large

                    except Exception as e:
                        st.error(f"Erreur lors de l'affichage de la visualisation '{original_key}': {e}")
                        if 'image_path' in locals(): # Vérifie si image_path a été défini
                             st.error(f"Chemin de l'image tenté : {image_path}")

                elif selected_display_title != "--- Choisir une visualisation ---":
                    st.warning("Erreur : impossible de trouver la clé originale pour le titre sélectionné.")

            else:
                st.info(f"Aucune visualisation disponible pour la catégorie '{category_name}' pour le moment.")


# --- Section Modélisation ---
elif current_choice == "⚙️ Modélisation":
    section_display_name = SECTION_ICONS.get(current_choice, current_choice)
    st.markdown(f"<h1 style='text-align: left;'>{section_display_name}</h1>", unsafe_allow_html=True)
    df_final_preview = pd.DataFrame() # Initialisation pour éviter les erreurs si le chargement échoue

    # --- Contenu pour la section Preprocessing (dans un expander fermé par défaut) ---
    # /!\ MODIFICATION: expanded=False /!\
    with st.expander("🛠️ Preprocessing des Données 🛠️", expanded=False):

        st.markdown(
    "<h2 style='color: #5533FF; text-align: center;'>🛠️ Preprocessing des Données 🛠️</h2>",
    unsafe_allow_html=True
) # <-- Parenthèse fermante ajoutée ici

        # Texte d'introduction
        st.write("""
        Avant de pouvoir entraîner nos modèles de Machine Learning, nous avons effectué un preprocessing minutieux des données. Cette étape cruciale garantit la qualité et la cohérence des données utilisées pour l'entraînement.
        """)

        # Section des jeux de données sources
        st.subheader("Afficher les jeux de données sources 📁")
        st.markdown("""
        *   **eco2mix-regional-cons-def.csv** : Consommation énergétique régionale par type de source.
        *   **temperature-quotidienne-regionale.csv** : Températures moyennes quotidiennes par région.
        *   **soutirages-regionaux-quotidiens-consolides-rpt.csv** : Consommation sectorielle régionale consolidée.
        *   **population - insee.csv** : Population régionale annuelle.
        *   **base_etablissement_par_tranche_effectif.csv** : Nombre d'entreprises par région.
        """)

        # Section des étapes clés du preprocessing
        st.markdown("<h3 style='color: #B19CD9;'>Étapes clés du preprocessing 🛠️⚙️</h3>", unsafe_allow_html=True)

        # --- Étape 1 ---
        st.markdown("1.  **Conversion et Nettoyage des Données**")
        st.markdown("""
        *   Conversion des colonnes de dates au format `datetime`.
        *   Suppression des colonnes redondantes ou non pertinentes (ex: 'Code INSEE région', 'Heure').
        """)

        # --- Étape 2 ---
        st.markdown("2.  **Création de Colonnes Dérivées et Agrégrations**")
        st.markdown("""
        *   Création de colonnes pour le jour, le mois et l'année à partir de la colonne date.
        *   Agrégation des données par jour et par région (moyennes, sommes).
        """)

        # --- Étape 3 ---
        st.markdown("3.  **Gestion des Valeurs Manquantes**")
        st.markdown("*   Remplacement des valeurs manquantes de température par la température moyenne mensuelle de la région concernée.")
        st.markdown("**Détails sur la gestion des valeurs manquantes** 🔍")
        st.markdown("*   Calcul de la température moyenne mensuelle par région.")
        st.markdown("*   Imputation des valeurs manquantes avec ces moyennes.")

        # --- Étape 4 ---
        st.markdown("4.  **Retraitement des Données Socio-Économiques**")
        st.markdown("*   Harmonisation des noms de régions.")
        st.markdown("*   Création d'une plage de dates complète pour les données de population.")

        # --- Étape 5 ---
        st.markdown("5.  **Harmonisation des Plages de Dates et Fusion des DataFrames**")
        st.markdown("*   Filtrage des données pour ne conserver que la période du 1er janvier 2019 au 31 janvier 2023.")
        st.markdown("*   Fusion des différents DataFrames en un seul DataFrame final, `df_final`, en utilisant la région et la date comme clés de jointure.")

        # --- Aperçu du DataFrame Final ---
        st.markdown("### Aperçu du DataFrame final (df_final)") # Utilisation de H3

        df_final_path = DF_FINAL_CSV_PATH
        try:
            # Essayer de lire avec sep=',' (comme dans l'exemple)
            df_final_preview = pd.read_csv(df_final_path, sep=',')
            st.dataframe(df_final_preview.head())
            print(f"--- df_final loaded successfully with sep=',' from {df_final_path}")
        except FileNotFoundError:
            st.error(f"Erreur : Le fichier du DataFrame final n'a pas été trouvé à l'emplacement : {df_final_path}")
            st.info(f"Assurez-vous que le fichier '{os.path.basename(df_final_path)}' existe dans le répertoire '{os.path.dirname(df_final_path)}'.")
            print(f"Error: df_final file not found at {df_final_path}")
            df_final_preview = pd.DataFrame()
        except Exception as e_comma:
            print(f"Warning: Failed reading df_final with sep=',': {e_comma}. Trying sep=';'...")
            # st.warning(f"Échec lecture df_final avec sep=',': {e_comma}. Essai avec sep=';'...")
            try:
                # Essayer avec sep=';' comme fallback
                df_final_preview = pd.read_csv(df_final_path, sep=';')
                st.dataframe(df_final_preview.head())
                print(f"--- df_final loaded successfully with sep=';' from {df_final_path}")
            except Exception as e_semi:
                 st.error(f"Une erreur est survenue lors du chargement ou de l'affichage du DataFrame final (avec sep=',' et sep=';') : {e_semi}")
                 st.warning(f"Vérifiez le format du fichier CSV ({df_final_path}), son séparateur et son encodage.")
                 print(f"Error: Failed reading df_final with both sep=',' and sep=';': {e_semi}")
                 traceback.print_exc()
                 df_final_preview = pd.DataFrame()

        # --- Matrice de Corrélation (reste à l'intérieur de l'expander) ---
        st.subheader("📊 Matrice de Corrélation")

        if os.path.exists(CORRELATION_MATRIX_IMAGE_PATH):
            col_img_left, col_img_center, col_img_right = st.columns([1, 2, 1])
            with col_img_center:
                st.image(CORRELATION_MATRIX_IMAGE_PATH, caption="Visualisation de la Matrice de Corrélation")
        else:
            st.warning(f"Image de la matrice de corrélation non trouvée : {CORRELATION_MATRIX_IMAGE_PATH}")
            st.info(f"Assurez-vous que le fichier '{os.path.basename(CORRELATION_MATRIX_IMAGE_PATH)}' existe dans le dossier '{os.path.dirname(CORRELATION_MATRIX_IMAGE_PATH)}'.")
            print(f"Warning: Correlation matrix image not found: {CORRELATION_MATRIX_IMAGE_PATH}")

        st.write("""
        **Conclusion :** La matrice de corrélation confirme l'importance des variables choisies pour la modélisation et met en évidence les relations attendues entre consommation énergétique, température et secteurs d'activité.
        """)

    # --- /!\ MODIFICATION : Expander pour la partie MODELISATION fermé par défaut /!\ ---
    # /!\ MODIFICATION: expanded=False /!\
    with st.expander("🤖 Modèle de Machine Learning 🤖", expanded=False): # <-- Expander fermé par défaut
        # Le titre H2 est maintenant redondant avec le titre de l'expander, nous le supprimons.
        # st.markdown("<h2 style='text-align: center; color: #5533FF;'>🤖 Modèle de Machine Learning 🤖</h2>", unsafe_allow_html=True)

        st.markdown(
    "<h2 style='color: #5533FF; text-align: center;'>🤖 Modèle de Machine Learning 🤖</h2>",
    unsafe_allow_html=True
)

        st.markdown("""
        Après avoir testé plusieurs modèles, le **Random Forest Regressor** s'est avéré le plus performant pour prédire la consommation énergétique. Il offre une excellente capacité de généralisation et un faible risque de surapprentissage.
        """)

        st.markdown("<h3 style='color: #B19CD9;'>Tableau Comparatif des Modèles 📊</h3>", unsafe_allow_html=True) # H3 pour sous-section

        st.write("Voici un tableau récapitulatif des performances des différents modèles testés :")

        # Création du DataFrame avec les données (basé sur l'image mais fourni dans le code précédent)
        # Assurez-vous que ces données sont correctes ou chargez-les si nécessaire
        model_data = {
            'Modèle': ['Random Forest', 'Decision Tree', 'LassoCV', 'RidgeCV', 'Linear Regression'],
            'MAE (Entraînement)': [2831.117357, 16726.290844, 36726.768638, 18794.077160, 18794.850116],
            'MSE (Entraînement)': [18071859, 516232825, 2269012402, 625388854, 625284276],
            'R² (Entraînement)': [0.998107, 0.945931, 0.762351, 0.934499, 0.934510],
            'MAE (Test)': [7702.511192, 16873.532846, 35941.477643, 18737.704528, 18733.597936],
            'MSE (Test)': [133642664, 532403034, 2194812149, 618958955, 618619517],
            'R² (Test)': [0.985936, 0.943973, 0.769033, 0.934865, 0.934901]
        }
        df_models_results = pd.DataFrame(model_data)
        df_models_results = df_models_results.set_index('Modèle') # Mettre le nom du modèle comme index

        # Fonction pour appliquer le style (surligner Random Forest)
        def highlight_rf(s):
            '''
            Highlights the Random Forest row with a specific background color.
            '''
            # Applique le style si le nom de l'index (Modèle) est 'Random Forest'
            return ['background-color: #5533FF; color: white;' if s.name == 'Random Forest' else '' for _ in s]

        # Appliquer le style et formater les nombres pour l'affichage
        # Ajustement du formatage pour correspondre à l'image (pas de séparateur de milliers)
        st.dataframe(
            df_models_results.style.apply(highlight_rf, axis=1).format({
                'MAE (Entraînement)': '{:.6f}',
                'MSE (Entraînement)': '{:.0f}',
                'R² (Entraînement)': '{:.6f}',
                'MAE (Test)': '{:.6f}',
                'MSE (Test)': '{:.0f}',
                'R² (Test)': '{:.6f}',
            }).format_index(escape="html"), # Pour s'assurer que l'index s'affiche correctement
            use_container_width=True # Utiliser toute la largeur disponible
        )

        st.markdown("""
        Comme le montre le tableau ci-dessus, le modèle **Random Forest** surpasse nettement les autres modèles testés en termes de R², de MAE et de MSE, tant sur les données d’entraînement que sur les données de test. Il offre donc la meilleure capacité prédictive pour la consommation énergétique dans notre cas d’étude.
        """)

        st.markdown("<h3 style='color: #B19CD9;'>Paramètres du modèle ⚙️</h3>", unsafe_allow_html=True)

        st.write("""
        Les hyperparamètres suivants ont été sélectionnés après optimisation avec **GridSearchCV** :
        """)

        st.markdown("""
        *   **n_estimators**: 125 🌳
            > Nombre d'arbres dans la forêt. Plus il y a d'arbres, plus le modèle est performant, mais plus il est lent à entraîner.
        *   **max_depth**: 20 🖊️
            > Profondeur maximale de chaque arbre. Une profondeur plus grande permet de capturer des relations plus complexes, mais augmente le risque de surapprentissage.
        *   **min_samples_split**: 2 ✂️
            > Nombre minimum d'échantillons requis pour diviser un nœud interne. Une valeur plus élevée permet d'éviter de créer des divisions trop spécifiques qui pourraient conduire à du surapprentissage.
        *   **min_samples_leaf**: 1 🌱
            > Nombre minimum d'échantillons requis dans un nœud feuille. Une valeur plus élevée permet d'éviter de créer des feuilles avec trop peu d'échantillons, ce qui pourrait conduire à du surapprentissage.
        *   **random_state**: 42 🎲
            > Graine du générateur de nombres aléatoires. Fixer cette valeur permet de garantir la reproductibilité des résultats.
        """)

        st.write("""
        **GridSearchCV** est une technique d'optimisation qui permet de tester différentes combinaisons d'hyperparamètres et de sélectionner la meilleure combinaison en fonction d'une métrique de performance (par exemple, le R²).
        """)

        st.divider() # Ajoute une ligne de séparation visuelle

        st.markdown("<h3 style='color: #B19CD9;'>Résultats du Modèle Random Forest 🎯</h3>", unsafe_allow_html=True)
        st.write("Voici les résultats obtenus avec le modèle Random Forest optimisé, sur les jeux d'entraînement et de test :")

        # Données exactes de l'image
        results_data_rf = {
            'Métrique': ['R²', 'MAE', 'MSE'],
            # Utiliser '.' pour les décimales en Python
            'Entraînement': [0.9976, 2991.7500, 19427856.7000],
            'Test': [0.9853, 7973.9100, 141255891.4400]
        }
        df_rf_results_specific = pd.DataFrame(results_data_rf).set_index('Métrique')

        # Affichage avec formatage pour correspondre à l'image de la demande
        # Utilisation de st.dataframe pour un rendu tabulaire standard
        # Formatage des nombres pour correspondre à la précision de l'image
        # Utilisation de la virgule comme séparateur de milliers et du point comme séparateur décimal
        st.dataframe(
            df_rf_results_specific.style.format({
                'Entraînement': '{:,.4f}'.format, # Format avec virgule milliers, point décimal, 4 décimales
                'Test': '{:,.4f}'.format
            }),
            use_container_width=True # Adapter à la largeur
        )

        st.markdown("<h3 style='color: #B19CD9;'>Interprétation des résultats 📊</h3>", unsafe_allow_html=True)
        # Texte exact de l'image avec formatage Markdown
        # Note: MSE values rounded to integer, R2 as percentage, MAE rounded to integer
        st.markdown("""
        Le modèle Random Forest offre des performances exceptionnelles, avec un coefficient de détermination (R²) de **99.76%** sur le jeu d'entraînement et de **98.53%** sur le jeu de test.
        Les erreurs absolues moyennes (MAE) sont de **2992 MW** et **7974 MW** respectivement, et les erreurs quadratiques moyennes (MSE) sont de **19 427 857 MW²** et **141 255 891 MW²**.
        Ces résultats indiquent que le modèle est capable de généraliser efficacement les données, tout en minimisant les erreurs de prédiction.
        """)

        st.markdown("<h3 style='color: #B19CD9;'>Graphique : Prédictions vs Réalité 📈</h3>", unsafe_allow_html=True)
        st.write("Le graphique suivant compare les prédictions du modèle Random Forest aux valeurs réelles sur le jeu de test :")

        # --- /!\ AJOUT DE L'IMAGE DEMANDÉE /!\ ---
        if os.path.exists(MODEL_PREDICTION_IMAGE_PATH):
            col_img_left, col_img_center, col_img_right = st.columns([1, 3, 1]) # Ratio pour centrer (colonne centrale plus large)
            with col_img_center:
                st.image(MODEL_PREDICTION_IMAGE_PATH, caption="Prédictions vs Réalité (Modèle Optimisé)")
        else:
            st.warning(f"Image 'Prédictions vs Réalité' non trouvée : {MODEL_PREDICTION_IMAGE_PATH}")
            st.info(f"Assurez-vous que le fichier '{os.path.basename(MODEL_PREDICTION_IMAGE_PATH)}' existe dans le dossier '{os.path.dirname(MODEL_PREDICTION_IMAGE_PATH)}'.")
            print(f"Warning: Model prediction image not found: {MODEL_PREDICTION_IMAGE_PATH}") # Log pour console
        # --- /!\ FIN AJOUT DE L'IMAGE /!\ ---

        # --- /!\ AJOUT DE LA CONCLUSION SUR LA MODELISATION /!\ ---
        st.divider() # Ajoute une ligne de séparation visuelle avant la conclusion finale

        # Utilise H3 avec la couleur standard des sous-sections ici et ajoute l'emoji ✅
        st.markdown("<h3 style='color: #B19CD9;'>Conclusion sur la Modélisation ✅</h3>", unsafe_allow_html=True)
        # Utilise st.markdown pour le texte, permettant le formatage si besoin plus tard
        st.markdown("""
        Le modèle Random Forest, avec les hyperparamètres optimisés, offre d'excellentes performances pour la prédiction de la consommation énergétique.
        Il présente une bonne capacité de généralisation et une grande précision. Il surpasse les autres modèles testés (régression linéaire, arbre de décision, Lasso, Ridge).
        """)
        # --- /!\ FIN AJOUT CONCLUSION /!\ ---

# --- /!\ FIN MODIFICATION : La partie Modélisation est maintenant dans l'expander /!\ ---

# --- Section Prédiction ---
elif current_choice == "🤖 Prédiction":
    section_display_name = SECTION_ICONS.get(current_choice, current_choice)
    st.markdown(f"<h1 style='text-align: left;'>{section_display_name}</h1>", unsafe_allow_html=True)
    st.write("Entrez les informations nécessaires pour prédire la consommation d'électricité, ou ajustez les suggestions de population et d'entreprises avec les curseurs.")
    st.markdown("---") # Ajout d'un séparateur

    # --- Vérification et chargement des artefacts ML ---
    # (Code inchangé ici - vérification pipeline, columns_info, regions_list)
    if pipeline is None or columns_info is None or regions_list is None:
        st.error("❌ Erreur critique : Impossible de charger les composants ML nécessaires (pipeline, infos colonnes, liste régions).")
        st.warning("Vérifiez les chemins des fichiers dans la configuration et les logs de démarrage pour les erreurs de chargement.")
        st.markdown(f"""
        Chemins des artefacts vérifiés :
        - Pipeline: `{BEST_PIPELINE_PATH}`
        - Infos Colonnes: `{COLUMNS_INFO_PATH}`
        - Régions: `{REGIONS_PATH}`
        """)
        st.stop()

    # --- Chargement des données contextuelles depuis df_final ---
    # (Code inchangé ici - chargement et vérification de df_final)
    df_final = get_df_final_data()
    if df_final is None:
        st.error(f"❌ Échec critique : Impossible de charger les données contextuelles depuis df_final.csv ({DF_FINAL_CSV_PATH}).")
        st.warning("La fonctionnalité de prédiction ne peut pas fournir de valeurs par défaut adaptées sans ces données.")
        st.info("Vérifiez que le fichier existe, qu'il est accessible et n'est pas corrompu.")
        st.stop()
    elif df_final.empty:
        st.warning("⚠️ Le fichier df_final.csv est vide ou n'a pas pu être traité correctement. Les valeurs par défaut génériques seront utilisées pour les champs de saisie.")
        context_available = False
        df_final = pd.DataFrame()
    else:
        required_context_cols = ['region', 'date', 'year', 'month', 'population', 'nb_total_entreprise', 'tmoy_degc']
        missing_context_cols = [col for col in required_context_cols if col not in df_final.columns]
        if missing_context_cols:
            st.warning(f"⚠️ Colonnes manquantes dans df_final pour le contexte : {', '.join(missing_context_cols)}. Les valeurs par défaut génériques seront utilisées.")
            context_available = False
        else:
            context_available = True
            try:
                if not pd.api.types.is_datetime64_any_dtype(df_final['date']):
                     df_final['date'] = pd.to_datetime(df_final['date'], errors='coerce')
                for col in ['population', 'nb_total_entreprise', 'tmoy_degc', 'year', 'month']:
                     if col in df_final.columns and not pd.api.types.is_numeric_dtype(df_final[col]):
                         df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
                df_final.dropna(subset=['date', 'region', 'year', 'month'], inplace=True)
                if df_final.empty:
                     st.warning("⚠️ df_final est devenu vide après nettoyage/conversion des types. Utilisation des valeurs par défaut génériques.")
                     context_available = False
            except Exception as e_conv:
                 st.warning(f"⚠️ Erreur lors de la vérification/conversion des types dans df_final : {e_conv}. Utilisation des valeurs par défaut génériques.")
                 context_available = False

    # --- Formulaire pour les entrées utilisateur ---
    st.subheader("📝 Paramètres d'entrée")
    col1, col2 = st.columns(2)

    with col1:
        # (Widgets Région et Date inchangés)
        selected_region = st.selectbox(
            "📍 Région :", options=regions_list, index=0, key="pred_region",
            help="Choisissez la région pour laquelle vous souhaitez faire une prédiction."
        )
        today = datetime.date.today()
        min_hist_date = df_final['date'].min().date() if context_available and not df_final.empty and 'date' in df_final.columns and not df_final['date'].isnull().all() else datetime.date(2019, 1, 1)
        max_pred_date = today + datetime.timedelta(days=730)
        selected_date = st.date_input(
            "🗓️ Date :", value=today, min_value=min_hist_date, max_value=max_pred_date, key="pred_date",
            help="Choisissez la date pour laquelle vous souhaitez faire une prédiction."
        )
        selected_month = selected_date.month
        selected_year = selected_date.year

        # Mettre la température ici aussi pour équilibrer les colonnes
        st.markdown("<br>", unsafe_allow_html=True) # Espace visuel
        temp_moyenne = st.number_input(
            "🌡️ Température Moyenne (°C) :",
            min_value=-20.0, max_value=45.0,
            value=15.0, # Garder une valeur par défaut simple ici, le contexte est calculé plus tard
            step=0.1, format="%.1f",
            key="pred_temp",
            help="Température moyenne prévue pour la journée."
        )


    # --- Calcul des valeurs suggérées (base pour les sliders) ---
    default_pop = 5000000 # Fallback
    default_companies = 300000 # Fallback
    context_pop = "Contexte non disponible"
    context_companies = "Contexte non disponible"
    # (La logique de calcul de default_pop, default_companies, context_pop, context_companies reste la même qu'avant)
    if context_available:
        df_region = df_final.loc[df_final['region'] == selected_region].copy()
        if not df_region.empty:
            min_year_data = int(df_region['year'].min())
            max_year_data = int(df_region['year'].max())
            target_year_socioeco = max(min_year_data, min(selected_year, max_year_data))

            # Population
            pop_col = df_region.loc[df_region['year'] == target_year_socioeco, 'population'].dropna()
            if not pop_col.empty and pd.api.types.is_numeric_dtype(pop_col):
                avg_pop = pop_col.mean()
                default_pop = int(round(avg_pop))
                context_pop = f"{avg_pop:,.0f} hab. (Moy. {selected_region} {target_year_socioeco})".replace(",", " ")
            else:
                context_pop = f"Donnée Pop. indisponible ({selected_region} {target_year_socioeco})"

            # Entreprises
            comp_col = df_region.loc[df_region['year'] == target_year_socioeco, 'nb_total_entreprise'].dropna()
            if not comp_col.empty and pd.api.types.is_numeric_dtype(comp_col):
                avg_comp = comp_col.mean()
                default_companies = int(round(avg_comp))
                context_companies = f"{avg_comp:,.0f} entr. (Moy. {selected_region} {target_year_socioeco})".replace(",", " ")
            else:
                context_companies = f"Donnée Entr. indisponible ({selected_region} {target_year_socioeco})"
        else:
             context_pop = f"Données contextuelles indisponibles pour {selected_region}"
             context_companies = context_pop


    # --- Affichage et Sliders pour Population et Entreprises ---
    with col2:
        st.markdown("👥 **Population (habitants)**")
        # Afficher la suggestion
        if context_available and not context_pop.startswith("Contexte non disponible") and not context_pop.startswith("Donnée"):
            st.caption(f"ℹ️ Suggestion historique : {context_pop}")
        else:
             st.caption(f"ℹ️ Suggestion historique : {default_pop:,.0f} (valeur générique)".replace(",", " "))

        # Slider pour ajustement en %
        pop_growth_percentage = st.slider(
            "Ajustement Croissance Pop. (%) :",
            min_value=-20.0,  # Permet une baisse de 10%
            max_value=20.0,   # MODIFIÉ: Permet une hausse de 10% max
            value=0.0,        # Défaut à 0% (pas d'ajustement)
            step=5.0,         # Pas de 0.5%
            format="%.1f%%",  # Affichage avec '%'
            key="pop_growth_slider",
            help=f"Ajustez la croissance par rapport à la suggestion basée sur l'historique ({context_pop if context_available else 'N/A'})."
        )
        # Calculer la valeur finale
        final_population = float(default_pop) * (1 + pop_growth_percentage / 100.0)
        final_population = int(round(final_population)) # Arrondir à l'entier le plus proche

        # Afficher la valeur finale utilisée (avec séparateur)
        st.markdown(f"↳ **Population finale estimée :** `{final_population:,.0f}`".replace(",", " "))

        st.markdown("---") # Séparateur visuel

        st.markdown("🏢 **Nombre Total d'Entreprises**")
        # Afficher la suggestion
        if context_available and not context_companies.startswith("Contexte non disponible") and not context_companies.startswith("Donnée"):
            st.caption(f"ℹ️ Suggestion historique : {context_companies}")
        else:
            st.caption(f"ℹ️ Suggestion historique : {default_companies:,.0f} (valeur générique)".replace(",", " "))

        # Slider pour ajustement en %
        company_growth_percentage = st.slider(
            "Ajustement Croissance Entr. (%) :",
            min_value=-20.0,  # Permet une baisse de 10%
            max_value=20.0,   # MODIFIÉ: Permet une hausse de 10% max
            value=0.0,        # Défaut à 0%
            step=5.0,         # Pas de 0.5%
            format="%.1f%%",  # Affichage avec '%'
            key="company_growth_slider",
            help=f"Ajustez la croissance par rapport à la suggestion basée sur l'historique ({context_companies if context_available else 'N/A'})."
        )
        # Calculer la valeur finale
        final_companies = float(default_companies) * (1 + company_growth_percentage / 100.0)
        final_companies = int(round(final_companies))

        # Afficher la valeur finale utilisée (avec séparateur)
        st.markdown(f"↳ **Nombre final d'entreprises :** `{final_companies:,.0f}`".replace(",", " "))


    # --- Bouton de Prédiction et Logique Associée ---
    st.markdown("---")
    if st.button("🔮 Lancer la Prédiction", key="predict_button", type="primary", use_container_width=True):

        # 1. Préparer le DataFrame d'entrée pour le modèle
        try:
            selected_datetime = datetime.datetime.combine(selected_date, datetime.datetime.min.time())

            # Extraire les features temporelles
            date_features = {
                'year': selected_datetime.year,
                'month': selected_datetime.month,
                'day': selected_datetime.day,
                'dayofweek': selected_datetime.weekday(),
                'dayofyear': selected_datetime.timetuple().tm_yday,
                'weekofyear': selected_datetime.isocalendar().week,
                'quarter': (selected_datetime.month - 1) // 3 + 1
            }

            # Construire le dictionnaire avec les données d'entrée
            # Utilise temp_moyenne du widget et les valeurs FINALES calculées par les sliders
            input_data = {
                'region': selected_region,
                'tmoy_degc': temp_moyenne,       # Valeur du widget température
                'population': float(final_population), # VALEUR FINALE du slider pop
                'nb_total_entreprise': float(final_companies), # VALEUR FINALE du slider entr
                **date_features
            }

            input_df = pd.DataFrame([input_data])

            # 2. S'assurer de l'ordre et de l'existence des colonnes
            if columns_info and 'original_features' in columns_info:
                expected_columns = columns_info['original_features']
            else:
                st.error("❌ Information cruciale manquante : Liste des 'original_features' non trouvée dans 'columns_info.json'.")
                st.stop()

            missing_cols_in_input = [col for col in expected_columns if col not in input_df.columns]
            if missing_cols_in_input:
                st.error(f"❌ Erreur interne : Colonnes manquantes lors de la création de l'entrée : {', '.join(missing_cols_in_input)}")
                st.stop()

            try:
                input_df_ordered = input_df[expected_columns]
            except KeyError as e:
                st.error(f"❌ Erreur lors de la réorganisation des colonnes : {e}.")
                st.stop()

            # 3. Faire la prédiction
            with st.spinner("🧠 Calcul de la prédiction en cours..."):
                prediction = pipeline.predict(input_df_ordered)

            # 4. Afficher le résultat
            predicted_value = prediction[0]
            st.markdown("---")
            st.subheader("✅ Résultat de la Prédiction")
            st.metric(
                label=f"Consommation Électrique Prédite pour {selected_region} le {selected_date.strftime('%d/%m/%Y')}",
                value=f"{predicted_value:,.2f} MW".replace(",", " "),
            )
            # Afficher un résumé des paramètres utilisés pour la prédiction
            st.markdown("Avec les paramètres finaux suivants :")
            st.markdown(f"- Température : `{temp_moyenne:.1f}°C`")
            st.markdown(f"- Population : `{final_population:,.0f}` (ajustement `{pop_growth_percentage:.1f}%`)".replace(',',' '))
            st.markdown(f"- Entreprises : `{final_companies:,.0f}` (ajustement `{company_growth_percentage:.1f}%`)".replace(',',' '))

            st.success("Prédiction calculée avec succès !")

        # (Gestion des erreurs inchangée)
        except FileNotFoundError as fnf_error:
            st.error(f"❌ Erreur de Fichier Non Trouvé : {fnf_error}")
        except KeyError as key_error:
            st.error(f"❌ Erreur de Clé (colonne manquante ?) : {key_error}")
            # ... (messages de debug potentiels)
        except ValueError as val_error:
             st.error(f"❌ Erreur de Valeur : {val_error}")
             # ... (messages de debug potentiels)
        except Exception as e:
            st.error(f"❌ Une erreur inattendue est survenue lors du processus de prédiction.")
            st.exception(e)
            st.info("Veuillez vérifier les logs de la console pour plus de détails.")
            # ... (messages de debug potentiels)


# --- Section Conclusion ---
elif current_choice == "📌 Conclusion":
    section_display_name = SECTION_ICONS.get(current_choice, current_choice)
    st.markdown(f"<h1 style='text-align: left;'>{section_display_name}</h1>", unsafe_allow_html=True)

    # --- Introduction de la Conclusion (existante) ---
    st.markdown("""
    Ce projet a permis d'explorer les dynamiques de consommation énergétique en France et d'établir des modèles prédictifs performants pour anticiper la demande énergétique. À travers des analyses approfondies des données historiques, nous avons identifié les facteurs clés influençant la consommation électrique, notamment les variations climatiques et les tendances démographiques.
    """)

    # --- Résultats Clés (existants) ---
    st.markdown("### 🌟 Résultats clés :")
    st.markdown("""
    *   **Exploration des données :**
        *   Identification des variations saisonnières de la consommation énergétique.
        *   Analyse des contributions des différentes sources d'énergie (nucléaire, renouvelables, etc.).
    *   **Modélisation et Prédictions :**
        *   Utilisation du modèle Random Forest pour prédire la consommation énergétique régionale avec un **R² de 98,5%** sur les données de test.
        *   Intégration de variables clés comme la température moyenne, la population et la région pour des prédictions précises.
    """)


    # --- /!\ AJOUT DU CONTENU DE L'IMAGE ICI /!\ ---

    # --- Points d'amélioration (depuis l'image) ---
    st.markdown("### 🔑 Points d'amélioration :") # Utilisation de l'emoji clé
    st.markdown("""
    *   **Intégration de nouvelles données :**
        *   Ajouter des variables comme les jours fériés, les vagues de froid/chaleur ou les événements économiques pour affiner les prédictions.
    *   **Adoption de modèles avancés :**
        *   Tester des modèles complexes tels que les réseaux neuronaux pour mieux capturer les non-linéarités dans les données.
    """)


    # --- Contribution pour la transition énergétique (depuis l'image) ---
    st.markdown("### 🌟 Contribution pour la transition énergétique :") # Utilisation de l'emoji étoile
    st.markdown("""
    Ce projet apporte des outils pour mieux comprendre et prévoir la consommation énergétique, contribuant ainsi à une gestion plus efficace des ressources.
    """)


    # --- Message final (depuis l'image) ---
    st.markdown( # CORRECTION: Ajout de la parenthèse fermante ici
        """
        <p style='text-align: center;'>
        🚀 Merci d'avoir suivi ce projet. Ensemble, avançons vers un futur énergétique durable ! 🚀
        </p>
        """,
        unsafe_allow_html=True
    )

    # --- /!\ FIN DE L'AJOUT DU CONTENU DE L'IMAGE /!\ ---


# =============================================================================
# --- 8. FOOTER --- (PROPOSITION 3 - AVEC SITE WEB)
# =============================================================================

# Définir les URLs et autres constantes
linkedin_url = "https://www.linkedin.com/in/jeremy-vanerpe/"
github_url = "https://github.com/JVEdata"
website_url = "https://jeremyvanerpe.fr/"
developer_name = "Jérémy VAN ERPE"

# Obtenir l'année actuelle dynamiquement
import datetime
current_year = datetime.date.today().year

# Construire le HTML du footer avec CSS intégré
footer_html = f"""
<style>
    /* Style pour le conteneur du footer */
    #app-footer-pro {{
        text-align: center;
        padding: 1.5em 0;
        margin-top: 3em;
        font-size: 0.9em;
        color: #CCCCCC;
        border-top: 1px solid #333333;
    }}

    /* Style pour les liens */
    #app-footer-pro a {{
        color: #A0DAFF;
        text-decoration: none;
        transition: color 0.2s ease;
    }}

    /* Style des liens au survol */
    #app-footer-pro a:hover {{
        color: #FFFFFF;
    }}

    /* Style pour les séparateurs */
    #app-footer-pro .separator {{
        margin: 0 0.7em;
        color: #555555;
    }}

    /* Style pour les icônes/emojis */
    #app-footer-pro .icon {{
        display: inline-block;
        margin-right: 0.35em;
        font-size: 1.1em;
        vertical-align: -0.1em;
    }}
</style>

<div id="app-footer-pro">
    <span>© {current_year} {developer_name}</span>
    <span class="separator">|</span>
    <span>Développé avec
        <span class="icon" role="img" aria-label="Éclair">⚡</span>Streamlit &
        <span class="icon" role="img" aria-label="Panda">🐼</span>Pandas
    </span>
    <span class="separator">|</span>
    <a href="{website_url}" target="_blank" rel="noopener noreferrer">
        <span class="icon" role="img" aria-label="Globe">🌐</span>Site Web
    </a>
    <span class="separator">|</span>
    <a href="{linkedin_url}" target="_blank" rel="noopener noreferrer">
        <span class="icon" role="img" aria-label="Lien">💼</span>LinkedIn
    </a>
    <span class="separator">|</span>
    <a href="{github_url}" target="_blank" rel="noopener noreferrer">
        <span class="icon" role="img" aria-label="Octopus GitHub">🐙</span>GitHub
    </a>
</div>
"""

# Assurez-vous que la bibliothèque streamlit est importée
#import streamlit as st # Déjà importé plus haut

# Afficher le footer en tant que HTML
st.markdown(footer_html, unsafe_allow_html=True)