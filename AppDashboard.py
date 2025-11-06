# ======================================================
# DASHBOARD STREAMLIT — Proyecto Accidentabilidad Vial Antioquia
# ======================================================

import streamlit as st
from pathlib import Path
from PIL import Image
import pandas as pd
import base64

# ======================================================
# CONFIGURACIÓN INICIAL
# ======================================================

st.set_page_config(
    page_title="Dashboard — Accidentabilidad Vial Valle de Aburrá",
    layout="wide",
    page_icon="🚦",
)

# Fondo blanco global
st.markdown("""
<style>
.stApp {
    background-color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# RUTAS Y CONFIGURACIÓN BASE
# ======================================================

BASE_DIR = Path(__file__).parent.resolve()
GRAF_DIR = BASE_DIR / "Graficas_Salida"
MODEL_DIR = BASE_DIR / "Modelo_Predict"
IMG_PATH = BASE_DIR / "Diseño" / "Map_portada.png"

# ======================================================
# FUNCIÓN PORTADA
# ======================================================

def get_base64_of_image(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def mostrar_portada():
    # Fondo con imagen extendida
    if IMG_PATH.exists():
        bg_base64 = get_base64_of_image(IMG_PATH)
    else:
        bg_base64 = ""

    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg_base64}");
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        background-size: 1664px 936px;
    }}
    .overlay {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 3rem 5rem;
        border-radius: 20px;
        max-width: 80%;
        margin: 8% auto;
        box-shadow: 0 4px 15px rgba(0,0,0,0.25);
        display: flex;
        align-items: center;
        justify-content: flex-end;
        min-height: 500px;
    }}
    .title {{
        font-family: "Times New Roman", serif;
        font-weight: bold;
        font-size: 36px;
        color: #1a1a1a;
        text-align: right;
        line-height: 1.4;
        width: 55%;
    }}
    .stButton>button {{
        min-width: 180px;
        height: 3rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 10px;
        background-color: #2e7d32;
        color: white;
        border: none;
    }}
    .stButton>button:hover {{
        background-color: #43a047;
        color: white;
    }}
    .button-row {{
        display: flex;
        justify-content: center;
        gap: 0.8rem;
        margin-top: 2rem;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Contenedor principal
    st.markdown("<div class='overlay'>", unsafe_allow_html=True)

    # Título a la derecha centrado
    st.markdown("""
    <div class='title'>
    Análisis de datos y modelado predictivo<br>
    sobre la accidentalidad vial<br>
    en el Valle de Aburrá<br>
    (2015–2019)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Botones centrados
    st.markdown("<div class='button-row'>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📊 Análisis Exploratorio"):
            st.session_state["pagina"] = "analisis"
    with col2:
        if st.button("🤖 Modelo Predictivo"):
            st.session_state["pagina"] = "modelo"
    with col3:
        if st.button("📈 Curvas y Resultados"):
            st.session_state["pagina"] = "curvas"
    with col4:
        if st.button("🧾 Resumen Ejecutivo"):
            st.session_state["pagina"] = "resumen"
    st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# FUNCIÓN DASHBOARD PRINCIPAL
# ======================================================

def mostrar_dashboard():
    @st.cache_data
    def load_images():
        return {
            "clase": Image.open(GRAF_DIR / "Accidentes_Clase_SVA.jpg"),
            "comuna": Image.open(GRAF_DIR / "Accidentes_Comuna_SVA.jpg"),
            "gravedad": Image.open(GRAF_DIR / "Accidentes_Gravedad_SVA.jpg"),
            "jornada": Image.open(GRAF_DIR / "Accidentes_Jornada_SVA.jpg"),
            "roc": Image.open(GRAF_DIR / "Curva_ROC_SVA.jpg"),
            "matriz": Image.open(GRAF_DIR / "Matriz_Confusion_SVA.jpg"),
        }

    @st.cache_data
    def load_data():
        return {
            "importancia": pd.read_csv(MODEL_DIR / "Importancia_Variables_RF.csv"),
            "predicciones": pd.read_csv(MODEL_DIR / "Predicciones_Nuevos_Accidentes.csv"),
            "resumen": (MODEL_DIR / "Resumen_Ejecutivo_Modelo.txt").read_text(encoding="utf-8")
        }

    imagenes = load_images()
    datos = load_data()

    st.title("🚦 Proyecto Accidentabilidad Vial Antioquia — Panel General")

    # Botón volver a portada
    if st.button("🏠 Volver a la Portada"):
        st.session_state["pagina"] = "inicio"

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Análisis Exploratorio",
        "🤖 Modelo Predictivo",
        "📈 Curvas y Matrices",
        "🧾 Resumen Ejecutivo"
    ])

    with tab1:
        st.header("Distribución de Accidentes por Variables Clave")
        col1, col2 = st.columns(2)
        with col1:
            st.image(imagenes["gravedad"])
            st.image(imagenes["jornada"])
        with col2:
            st.image(imagenes["clase"])
            st.image(imagenes["comuna"])

    with tab2:
        st.header("Importancia de Variables — Modelo Random Forest")
        st.dataframe(datos["importancia"], use_container_width=True)
        st.subheader("Predicciones Simuladas")
        st.dataframe(datos["predicciones"], use_container_width=True)

    with tab3:
        st.header("Evaluación del Modelo de Clasificación")
        col3, col4 = st.columns(2)
        with col3:
            st.image(imagenes["roc"])
        with col4:
            st.image(imagenes["matriz"])

    with tab4:
        st.header("📄 Resumen Ejecutivo del Proyecto")
        st.text(datos["resumen"])


# ======================================================
# CONTROL DE NAVEGACIÓN
# ======================================================

if "pagina" not in st.session_state:
    st.session_state["pagina"] = "inicio"

if st.session_state["pagina"] == "inicio":
    mostrar_portada()
else:
    mostrar_dashboard()
