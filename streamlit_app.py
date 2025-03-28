import streamlit as st
import pandas as pd
from PIL import Image
import io
import base64
import numpy as np
import gdown
import os

# =============================================================================
# --- 1. CONFIGURATION ---
# =============================================================================

# Google Drive file IDs
GOOGLE_DRIVE_CSV_FILE_ID = '1hvEXS7ABSGUh45QHvoR1aCKzJzsQ0S4B'
GOOGLE_DRIVE_SOUTIRAGE_FILE_ID = '1vIsX-TemlEYBTH9dNA4vaEY_6lcYiX6j' # Soutirage Dataset
GOOGLE_DRIVE_IMG_FILE_ID = '1oVLP_z33SwbFF3rSoqFz0FmqYFbyhN-9'  # Image ID
GOOGLE_DRIVE_POPULATION_FILE_ID = '1XPmqG7k79mcDX2U93FPryrs9iUXVQwK0' # Population Dataset

SECTION_ICONS = {
    "👋 Introduction": "👋 Introduction",
    "🔎 Exploration des données": "🔎 Exploration des données",
    "📊 Data Visualisation": "📊 Data Visualisation",
    "⚙️ Modélisation": "⚙️ Modélisation",
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

with st.sidebar:
    st.markdown("<h1 style='color: #5533FF;'>📚 Sommaire</h1>", unsafe_allow_html=True)
    choix = st.radio(
        "Aller vers 👇",
        list(SECTION_ICONS.keys()),
        key='choix_radio',
        format_func=lambda x: SECTION_ICONS.get(x, x)  # Use icons for options
    )
    st.session_state.choix = choix
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("🔗 [Contactez-nous](https://www.linkedin.com/in/jeremyvanerpe)")

# =============================================================================
# --- 4. DATA LOADING FUNCTIONS ---
# =============================================================================

@st.cache_data
def load_data(file_id):
    """Loads data from a CSV file hosted on Google Drive."""
    output_path = f"temp_data_{file_id}.csv"
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        actual_output_path = gdown.download(url, output_path, quiet=True, fuzzy=True)

        if not actual_output_path or not os.path.exists(actual_output_path):
            st.error(f"Failed to download CSV file (ID: {file_id}). Check ID and sharing permissions.")
            return {}

        try:
            eco2mix_df = pd.read_csv(actual_output_path, sep=';', encoding='utf-8')
        except UnicodeDecodeError:
            st.warning("UTF-8 decoding failed, trying latin-1 for CSV.")
            eco2mix_df = pd.read_csv(actual_output_path, sep=';', encoding='latin-1')

        eco2mix_df = eco2mix_df.replace('ND', np.nan)
        
        return eco2mix_df

    except Exception as e:
        st.error(f"Error loading or processing CSV data: {e}")
        return None

    finally:
        if 'actual_output_path' in locals() and os.path.exists(actual_output_path):
            try:
                os.remove(actual_output_path)
            except OSError as e:
                st.warning(f"Could not remove temporary file {actual_output_path}: {e}")

@st.cache_data
def load_soutirage_data(file_id):
    """Loads the Soutirage data from Google Drive."""
    output_path = f"temp_soutirage_{file_id}.csv"
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        actual_output_path = gdown.download(url, output_path, quiet=True, fuzzy=True)

        if not actual_output_path or not os.path.exists(actual_output_path):
            st.error(f"Failed to download Soutirage CSV file (ID: {file_id}). Check ID and sharing permissions.")
            return None

        try:
            soutirage_df = pd.read_csv(actual_output_path, sep=';', encoding='utf-8')
        except UnicodeDecodeError:
            st.warning("UTF-8 decoding failed, trying latin-1 for Soutirage CSV.")
            soutirage_df = pd.read_csv(actual_output_path, sep=';', encoding='latin-1')

        return soutirage_df

    except Exception as e:
        st.error(f"Error loading or processing Soutirage CSV data: {e}")
        return None

    finally:
        if 'actual_output_path' in locals() and os.path.exists(actual_output_path):
            try:
                os.remove(actual_output_path)
            except OSError as e:
                st.warning(f"Could not remove temporary Soutirage file {actual_output_path}: {e}")

@st.cache_data
def load_population_data(file_id):
    """Loads population data from Google Drive."""
    output_path = f"temp_population_{file_id}.csv"
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        actual_output_path = gdown.download(url, output_path, quiet=True, fuzzy=True)

        if not actual_output_path or not os.path.exists(actual_output_path):
            st.error(f"Failed to download Population CSV file (ID: {file_id}). Check ID and sharing permissions.")
            return None

        try:
            population_df = pd.read_csv(actual_output_path, encoding='latin-1') # force latin-1 encoding
        except Exception as e:
            st.error(f"Error loading or processing Population CSV data: {e}")
            return None

        return population_df

    except Exception as e:
        st.error(f"Error loading or processing Population CSV data: {e}")
        return None

    finally:
        if 'actual_output_path' in locals() and os.path.exists(actual_output_path):
            try:
                os.remove(actual_output_path)
            except OSError as e:
                st.warning(f"Could not remove temporary Population file {actual_output_path}: {e}")

@st.cache_data
def load_and_process_image(image_file_id):
    """Loads and processes an image from Google Drive."""
    output_path = f"temp_image_{image_file_id}"
    image_data = {"img_str": None, "mime_type": "image/png"}

    try:
        url = f'https://drive.google.com/uc?id={image_file_id}'
        actual_output_path = gdown.download(url, output_path, quiet=True, fuzzy=True)

        if not actual_output_path or not os.path.exists(actual_output_path):
            st.error(f"Failed to download image (ID: {image_file_id}). Check ID and permissions.")
            return image_data

        image = Image.open(actual_output_path)

        # Resize
        width, height = image.size
        max_width = 400
        if width > max_width:
            ratio = max_width / width
            height = int(height * ratio)
            width = max_width
        image = image.resize((width, height))

        # Convert image to base64 string
        buffered = io.BytesIO()
        save_format = image.format or 'PNG'  # Default to PNG
        save_format = save_format.upper()
        if save_format == 'JPG':
            save_format = 'JPEG'

        if save_format not in ['JPEG', 'PNG', 'GIF', 'WEBP', 'BMP']:
            st.warning(f"Original image format '{save_format}' not directly supported, converting to PNG.")
            save_format = 'PNG'
            if image.mode in ['P', 'RGBA']:
                image = image.convert('RGB')  # Or RGBA if you need transparency

        image.save(buffered, format=save_format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        mime_type = Image.MIME.get(save_format, 'image/png')
        image_data["img_str"] = img_str
        image_data["mime_type"] = mime_type
        return image_data

    except FileNotFoundError:
        st.error(f"Temporary image file not found after download: {output_path}")
        return image_data
    except Exception as e:
        st.error(f"Error loading or processing the image: {e}")
        return image_data
    finally:
        if 'actual_output_path' in locals() and os.path.exists(actual_output_path):
            try:
                os.remove(actual_output_path)
            except OSError as e:
                st.warning(f"Could not remove temporary image file {actual_output_path}: {e}")

# =============================================================================
# --- 5. MAIN APP LOGIC ---
# =============================================================================

eco2mix_df = load_data(GOOGLE_DRIVE_CSV_FILE_ID)
soutirage_df = load_soutirage_data(GOOGLE_DRIVE_SOUTIRAGE_FILE_ID)
population_df = load_population_data(GOOGLE_DRIVE_POPULATION_FILE_ID)

datasets = {}

if eco2mix_df is not None:
    datasets["📁 Dataset Eco2mix (Fichier de base du projet)"] = eco2mix_df

if soutirage_df is not None:
    datasets["⚡ Soutirages régionaux quotidiens consolidés"] = soutirage_df

if population_df is not None:
    datasets["👪 Population - Insee"] = population_df

# Handle case where session state might not have 'choix' initially
if 'choix' not in st.session_state:
    st.session_state.choix = list(SECTION_ICONS.keys())[0] # Default to first item

current_choice = st.session_state.choix

# =============================================================================
# --- 6. PAGE CONTENT ---
# =============================================================================

if current_choice == "👋 Introduction":

    # Introduction Section Content
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
    les dynamiques de production et de consommation, particulièrement marquée par la prédominance du nucléaire et la croissance des énergies renouvelables. Ce projet vise à répondre aux
    enjeux stratégiques suivants :
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

    image_data = load_and_process_image(GOOGLE_DRIVE_IMG_FILE_ID)
    if image_data and image_data["img_str"]:
        img_str = image_data["img_str"]
        mime_type = image_data["mime_type"]
        html_code = f"""
        <div style="display: flex; justify-content: center; margin-top: 20px; margin-bottom: 10px;">
            <img src="data:{mime_type};base64,{img_str}" alt="Énergie éolienne" style="max-width: 400px; border-radius: 8px;">
        </div>
        <p style="text-align: center; font-style: italic;">Énergie éolienne</p>
        """
        st.markdown(html_code, unsafe_allow_html=True)

    if "show_exploration_message" not in st.session_state:
        st.session_state.show_exploration_message = False

    if st.button("🚀 Commencer l'exploration"):
        st.session_state.show_exploration_message = True
        st.session_state.choix = "🔎 Exploration des données" # Change state
        st.rerun() # Rerun the script to reflect the state change

    if st.session_state.show_exploration_message:
        st.success("Vous êtes prêt à découvrir les données énergétiques ! 🌍")

elif current_choice == "🔎 Exploration des données":

    # Data Exploration Section Content
    st.markdown("<h1 style='text-align: left;'>🔎 Exploration des Données</h1>", unsafe_allow_html=True)
    st.subheader("Sélectionnez un dataset à explorer")

    if not datasets:
        st.error("Le chargement d'au moins un dataset a échoué. Impossible de continuer l'exploration.")
    else:
        dataset_names = list(datasets.keys())
        selected_dataset_name = st.selectbox(
            "Choisissez un dataset :",
            dataset_names
        )

        if selected_dataset_name in datasets:
            df = datasets[selected_dataset_name]  # Renamed to 'df' for brevity

            if selected_dataset_name == "📁 Dataset Eco2mix (Fichier de base du projet)":
                st.markdown("<hr style='border: 1px solid #444;'>", unsafe_allow_html=True)
                st.markdown("""<h3 style='text-align: left;'> 📝 Aperçu de Dataset Eco2mix """, unsafe_allow_html=True)
                # You might want to add a more detailed description here
                st.write("""Le dataset Eco2mix est le cœur de ce projet. Il regroupe des données régionales et nationales sur la production et la consommation électrique en France, collectées à des intervalles de 30 minutes depuis 2013. Cette granularité permet d'identifier les variations saisonnières, les anomalies, et les tendances de long terme.""")
                st.write("Voici un aperçu des 5 premières lignes de la DataFrame :")
                st.dataframe(df.head())

                # --- Analysis of Missing Values Section ---
                st.markdown("<hr style='border: 1px solid #444;'>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: left;'>❓ Analyse des valeurs manquantes</h3>", unsafe_allow_html=True)
                st.write("Les valeurs manquantes dans le dataset sont principalement dues à plusieurs facteurs:")
                st.markdown("""
                <ul>
                    <li>Nucléaire : Certaines régions ne produisent pas d'énergie nucléaire, ce qui explique des cellules vides.</li>
                    <li>Stockage/Déstockage batterie, Éolien terrestre/offshore, TCO et TCH : Ces colonnes n'ont commencé à être renseignées qu'à une date ultérieure, augmentant les NaN pour les périodes antérieures. Pour garantir une analyse robuste, nous veillerons à sélectionner une plage de dates commune pour les analyses impliquant ces colonnes.</li>
                </ul>
                """, unsafe_allow_html=True)

                st.write("Nombre de valeurs manquantes par colonne:")

                # Calculate missing values per column
                missing_values = df.isnull().sum()
                missing_values_df = pd.DataFrame({'Column': missing_values.index, 'Missing Count': missing_values.values})

                # Display the table using st.dataframe
                st.dataframe(missing_values_df)

                # --- "Filter Data" Button ---
                st.markdown("<hr style='border: 1px solid #444;'>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: left;'>🗓️ Filtrer les données par région, jour, mois et année", unsafe_allow_html=True)


                # Ensure 'Région' column exists
                if 'Région' not in df.columns:
                    st.error("La colonne 'Région' est manquante dans le dataset Eco2mix.")
                else:
                    # Check if 'Région' column has data
                    if df['Région'].isnull().all():
                            st.warning("La colonne 'Région' existe mais ne contient aucune donnée.")
                            regions = []
                    else:
                            regions = sorted(df['Région'].dropna().unique())

                    if not regions:
                        st.info("Aucune région disponible pour le filtrage.")
                    else:
                        selected_region = st.selectbox("Sélectionner une Région:", regions)

                        # Ensure 'Date' column exists and convert it to datetime objects if it's not already
                        if 'Date' not in df.columns:
                            st.error("La colonne 'Date' est manquante dans le dataset Eco2mix.")
                        else:
                            try:
                                # Convert to datetime only if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                                if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # Coerce invalid dates to NaT

                                # Drop rows where Date conversion failed
                                df.dropna(subset=['Date'], inplace=True)

                                if df['Date'].empty:
                                    st.warning("Aucune date valide trouvée après conversion.")
                                else:
                                    min_date = df['Date'].min().date()
                                    max_date = df['Date'].max().date()
                                    # Ensure default value is within bounds
                                    default_date = min_date if min_date <= max_date else max_date
                                    selected_date_input = st.date_input("Sélectionner une Date:",
                                                                        min_value=min_date,
                                                        max_value=max_date,
                                                        value=default_date) # Use corrected default

                                    # Convert selected_date_input (which is datetime.date) for comparison
                                    selected_date = pd.to_datetime(selected_date_input).date()


                                    # Filter the DataFrame
                                    # Ensure 'Consommation (MW)' exists before filtering/summing
                                    if 'Consommation (MW)' not in df.columns:
                                        st.error("La colonne 'Consommation (MW)' est manquante.")
                                        total_consumption = 0 # Or handle differently
                                        filtered_df = pd.DataFrame() # Empty df
                                    else:
                                        # Convert 'Consommation (MW)' to numeric, coercing errors
                                        df['Consommation (MW)'] = pd.to_numeric(df['Consommation (MW)'], errors='coerce')

                                        filtered_df = df[(df['Région'] == selected_region) & (df['Date'].dt.date == selected_date)].copy() # Use .copy() to avoid SettingWithCopyWarning

                                        # Calculate total consumption on the filtered (and numeric) data
                                        total_consumption = filtered_df['Consommation (MW)'].sum() # NaNs are automatically skipped by sum()

                                        # Display filtered data if needed
                                        # st.write("Données filtrées :")
                                        # st.dataframe(filtered_df)

                                        # Format consumption AFTER calculation
                                        # Use French locale formatting if possible, otherwise manual replacement
                                        try:
                                            # Attempt locale-based formatting (might require locale setup on the system/container)
                                            import locale
                                            try:
                                                locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8') # Try common French locale
                                            except locale.Error:
                                                locale.setlocale(locale.LC_ALL, 'French_France.1252') # Try Windows French locale
                                            formatted_consumption = locale.format_string("%.0f", total_consumption, grouping=True)
                                        except (ImportError, locale.Error):
                                            # Fallback manual formatting
                                            formatted_consumption = "{:,.0f}".format(total_consumption).replace(",", " ").replace(".", ",") # Use space for thousands

                                        st.markdown(f"""
    <div style='background-color: #1E1E2F; padding: 20px; border-radius: 8px; text-align: center;'>
        <h4 style='font-weight: bold; color: white; margin-bottom: 10px;'>💡 Consommation totale d'électricité ⚡</h4>
        <p style='color: #BBBBBB; font-size: 16px;'>
            <span style='font-weight: bold; font-size: 18px; color: #FF8C00; text-shadow: 0px 0px 10px rgba(255, 140, 0, 0.6);'>
                📅 <span style='color: #FFD700; font-size: 20px;'>le {selected_date_input.strftime('%d/%m/%Y')} en {selected_region}</span> 🌍
            </span>
        </p>
        <h2 style='color: #5533FF; font-weight: bold; letter-spacing: 1px;'>
            {formatted_consumption} MW ⚡ 🔋
        </h2>
    </div>
""", unsafe_allow_html=True)






                            except Exception as e: # Catch broader exceptions during date processing
                                st.error(f"Erreur lors du traitement de la colonne 'Date' ou du filtrage : {e}")
            elif selected_dataset_name == "⚡ Soutirages régionaux quotidiens consolidés":
                st.markdown("<hr style='border: 1px solid #444;'>", unsafe_allow_html=True)
                st.markdown("""<h3 style='text-align: left;'> 📝 Aperçu de Dataset Soutirages régionaux quotidiens consolidés """, unsafe_allow_html=True)
                st.write("""Ce dataset fournit les soutirages quotidiens consolidés au niveau régional, représentant l'électricité prélevée sur le réseau pour répondre aux besoins des consommateurs finaux. Les colonnes horaires (par exemple : "00h00", "01h30") indiquent les volumes soutirés en MW pour chaque demi-heure.""")
                st.write("Voici un aperçu des 5 premières lignes de la DataFrame :")
                st.dataframe(df.head())

                # --- Analysis of Missing Values Section ---
                st.markdown("<hr style='border: 1px solid #444;'>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: left;'>❓ Analyse des valeurs manquantes</h3>", unsafe_allow_html=True)
                st.write("Certaines colonnes horaires (de \"00h30\" à \"23h30\") présentaient des valeurs manquantes, mais celles-ci ont été supprimées lors du prétraitement des données. Actuellement, des colonnes horaires présentent encore de faibles taux de valeurs manquantes, représentant moins de 0,003 % des enregistrements. Ces valeurs ont été corrigées par leurs suppressions.")
                st.write("Nombre de valeurs manquantes par colonne:")

                # Calculate missing values per column
                missing_values = df.isnull().sum()
                missing_values_df = pd.DataFrame({'Column': missing_values.index, 'Missing Count': missing_values.values})
                missing_values_df = missing_values_df[missing_values_df['Missing Count'] > 0]

                # Display the table using st.dataframe
                st.dataframe(missing_values_df)

            elif selected_dataset_name == "👪 Population - Insee":
                st.markdown("<hr style='border: 1px solid #444;'>", unsafe_allow_html=True)
                st.markdown("""<h3 style='text-align: left;'> 📝 Aperçu de Population - Insee """, unsafe_allow_html=True)
                st.write("""Voici un aperçu des données provenant de l'Insee. Ces données fournissent des informations sur la population totale des différentes régions françaises, collectées sur plusieurs années. Cela permet d'explorer l'impact démographique sur la consommation énergétique.""")
                st.write("Voici un aperçu des 5 premières lignes de la DataFrame :")
                st.dataframe(df.head())

                # --- Analysis of Missing Values Section ---
                st.markdown("<hr style='border: 1px solid #444;'>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: left;'>❓ Analyse des valeurs manquantes</h3>", unsafe_allow_html=True)

                missing_values = df.isnull().sum()
                if missing_values.sum() == 0:
                    st.write("Le jeu de données est complet, sans valeurs manquantes. Cela garantit une qualité élevée et permet de l'utiliser directement pour des analyses sans prétraitement supplémentaire.")
                   
                
                # Prepare the data for the table
                missing_data = {'Column': missing_values.index, 'Missing Count': missing_values.values}
                missing_df = pd.DataFrame(missing_data)

                # Format 'Missing Count' to display as an integer (no decimal points)
                missing_df['Missing Count'] = missing_df['Missing Count'].astype(int)

                st.dataframe(missing_df)

                # --- Population by Region and Year Section ---
                st.markdown("<hr style='border: 1px solid #444;'>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: left;'>🏘️ Nombre d'habitants par région et par année</h3>", unsafe_allow_html=True)

                # Extract unique regions and years from the population dataset
                available_regions = sorted(df['Régions'].unique()) # Assuming 'Régions' is the correct column name

                #Get the year columns based on column index
                year_columns = df.columns[1:].tolist()
                available_years = sorted(year_columns)

                # Create select boxes for region and year
                selected_region_population = st.selectbox("Sélectionner une région :", available_regions)
                selected_year_population = st.selectbox("Sélectionner une année :", available_years)

                # Find the population for the selected region and year
                try:
                    selected_population = df[df['Régions'] == selected_region_population][selected_year_population].values[0] #
                    year_only = selected_year_population.split('/')[2] #getting the year
                    st.write(f"Nombre d'habitants en {year_only} en {selected_region_population} : {selected_population}")
                except IndexError:
                    st.warning(f"Aucune donnée disponible pour {selected_region_population} en {selected_year_population}.")
                except KeyError as e:
                    st.error(f"La colonne '{e}' n'existe pas dans le dataset.")
                except Exception as e:
                    st.error(f"Une erreur s'est produite : {e}")


            else:
                st.subheader(f"Aperçu du jeu de données : {selected_dataset_name}")
                st.dataframe(df.head())
        else:
            st.warning(f"Dataset '{selected_dataset_name}' could not be loaded or does not exist.")

elif current_choice in ["📊 Data Visualisation", "⚙️ Modélisation", "🤖 Prédiction", "📌 Conclusion"]:

    # Under Development Sections
    # Extract title robustly, handling potential missing space
    section_title = SECTION_ICONS.get(current_choice, current_choice).split(' ', 1)[-1] if ' ' in SECTION_ICONS.get(current_choice, current_choice) else current_choice
    st.title(section_title)
    st.info("🚧 Section en cours de développement 🚧")
    st.write("Revenez bientôt pour découvrir les visualisations, les modèles et les prédictions !")

# =============================================================================
# --- 7. FOOTER ---
# =============================================================================

st.markdown("<hr style='margin-top: 3rem; margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; font-size: 0.9em; color: #777;'>
        Créé avec ❤️ par <a href='https://www.linkedin.com/in/jeremyvanerpe' target='_blank' style='color: #5533FF; text-decoration: none;'>Jérémy VAN ERPE</a> |
        Code source sur <a href='https://github.com/vanerpe' target='_blank' style='color: #5533FF; text-decoration: none;'>GitHub</a> ⚡
    </div>
    """,
    unsafe_allow_html=True
)