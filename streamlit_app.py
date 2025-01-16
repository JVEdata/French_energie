import streamlit as st
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Consommation d'Électricité en France",
    page_icon="⚡",
    layout="wide"
)

# Sidebar Menu
with st.sidebar:
    st.markdown("<h1 style='color: #5533FF;'>📚 Sommaire</h1>", unsafe_allow_html=True)
    choix = st.radio(
        "Aller vers 👇",
        ["👋 Introduction", "🔎 Exploration des données", "📊 Data Visualisation", "⚙️ Modélisation", "🤖 Prédiction", "📌 Conclusion"]
    )
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("🔗 [Contactez-nous](https://www.linkedin.com/in/jeremyvanerpe)")

# Section d'introduction
if choix == "👋 Introduction":
    st.markdown("<h1 style='text-align: center; color: #5533FF;'>👋 Bienvenue sur notre projet de consommation d'électricité en France ⚡</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='text-align: center; font-size: 18px;'>
        La consommation d'électricité est un enjeu majeur dans la transition énergétique. Ce projet explore les données françaises
        afin de mieux comprendre les tendances de consommation, visualiser les variations saisonnières, 
        et prévoir la demande énergétique future. 🌱
        </p>
        """,
        unsafe_allow_html=True
    )
    st.write("""
    💡 **Pourquoi ce projet ?**  
    La transition énergétique est l'un des plus grands défis du XXIe siècle. La consommation d'électricité varie selon les saisons, les jours de la semaine, et même les heures de la journée.  
    Ce projet se concentre sur **l'analyse des données de consommation d'électricité en France** afin d'identifier des tendances, des anomalies, et d'établir des modèles prédictifs qui pourraient être utiles aux acteurs du secteur énergétique.

    🔎 **Objectifs du projet :**  
    - Explorer les données de consommation d'électricité en France
    - Créer des visualisations interactives des tendances énergétiques
    - Construire des modèles de prédiction basés sur les données historiques
    - Fournir des insights utiles pour la transition énergétique et la gestion de la demande
    """)
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/France_energy_mix.png/800px-France_energy_mix.png",
        caption="Mix énergétique en France - Source: Wikipedia"
    )
    if st.button("🚀 Commencer l'exploration"):
        st.write("Vous êtes prêt à découvrir les données énergétiques ! 🌍")

# Section d'exploration des données
elif choix == "🔎 Exploration des données":
    st.title("🔎 Exploration des Données")
    
    # Chargement des données
    file_path = '/workspaces/French_energie/eco2mix-regional-cons-def.csv'
    data = pd.read_csv(file_path, sep=';')

    # Aperçu des données
    st.subheader("Aperçu des données")
    st.write(data.head())

    # Informations sur les colonnes
    st.subheader("Informations sur les colonnes")
    st.write("Nombre de colonnes : ", len(data.columns))
    st.write("Colonnes disponibles :", list(data.columns))

    # Statistiques descriptives
    st.subheader("Statistiques descriptives")
    st.write(data.describe())

    # Comptage des valeurs manquantes
    st.subheader("Valeurs manquantes")
    st.write(data.isnull().sum())

    # Filtres interactifs
    st.subheader("Filtrer les données par région et date")
    regions = data['Région'].unique()
    selected_region = st.selectbox("Sélectionnez une région", regions)
    filtered_data = data[data['Région'] == selected_region]

    dates = filtered_data['Date'].unique()
    selected_date = st.selectbox("Sélectionnez une date", dates)
    filtered_data_by_date = filtered_data[filtered_data['Date'] == selected_date]

    st.write("Données filtrées :", filtered_data_by_date)

    # Graphique interactif - Consommation par région
    st.subheader("📊 Consommation d'électricité par source")
    fig = px.line(
        filtered_data_by_date,
        x='Heure',
        y=['Consommation (MW)', 'Thermique (MW)', 'Nucléaire (MW)', 'Eolien (MW)', 'Solaire (MW)', 'Hydraulique (MW)'],
        title=f"Consommation d'électricité le {selected_date} dans la région {selected_region}"
    )
    st.plotly_chart(fig)

    # Histogramme interactif - Distribution de la consommation
    st.subheader("📈 Distribution de la consommation")
    fig_hist = px.histogram(
        filtered_data_by_date,
        x='Consommation (MW)',
        nbins=30,
        title=f"Distribution de la consommation le {selected_date} dans la région {selected_region}"
    )
    st.plotly_chart(fig_hist)

    # Résumé dynamique des tendances
    st.subheader("🔍 Résumé des tendances")
    moyenne_consommation = filtered_data_by_date['Consommation (MW)'].mean()
    st.write(f"💡 La consommation moyenne le {selected_date} dans la région {selected_region} est de {moyenne_consommation:.2f} MW.")

# Section Data Visualisation
elif choix == "📊 Data Visualisation":
    st.title("📊 Data Visualisation")
    st.write("""
    Explorez les visualisations interactives qui illustrent les tendances de consommation d'électricité.
    """)

# Section Modélisation
elif choix == "⚙️ Modélisation":
    st.title("⚙️ Modélisation")
    st.write("""
    Ici, nous allons construire et tester un modèle de prédiction basé sur les données de consommation passées.
    """)

# Section Prédiction
elif choix == "🤖 Prédiction":
    st.title("🤖 Prédiction")
    st.write("""
    Découvrez les prédictions de la consommation d'électricité future basées sur notre modèle.
    """)

# Section Conclusion
elif choix == "📌 Conclusion":
    st.title("📌 Conclusion")
    st.write("""
    Résumons les principales découvertes du projet et examinons les perspectives pour la consommation d'énergie en France.
    """)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center;'>
        Créé avec ❤️ par <a href='https://www.linkedin.com/in/jeremyvanerpe' target='_blank'>Jérémy VAN ERPE</a> |
        Suivez-moi sur <a href='https://github.com/vanerpe' target='_blank'>GitHub</a> ⚡
    </div>
    """,
    unsafe_allow_html=True
)
