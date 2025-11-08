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
IMG_PATH = BASE_DIR / "Diseño" / "Map_portada.jpeg"

# ======================================================
# FUNCIÓN PORTADA MEJORADA - TODO DENTRO DEL OVERLAY
# ======================================================

def get_base64_of_image(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def mostrar_portada():
    # Fondo con imagen fija y estática
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
        background-size: cover;
        background-color: #000000;
    }}
    
    /* Eliminar padding por defecto de Streamlit */
    .main .block-container {{
        padding-top: 0;
        padding-bottom: 0;
        padding-left: 0;
        padding-right: 0;
    }}
    
    /* Overlay principal que contiene TODO */
    .overlay-container {{
        background-color: rgba(255, 255, 255, 0.92);
        padding: 4rem 3rem;
        border-radius: 20px;
        max-width: 85%;
        margin: 10vh auto;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        min-height: 500px;
        position: relative;
        z-index: 1;
    }}
    
    /* Contenedor del contenido interno */
    .content-wrapper {{
        display: flex;
        flex-direction: column;
        height: 100%;
        justify-content: space-between;
    }}
    
    /* Título mejorado */
    .title {{
        font-family: "Times New Roman", serif;
        font-weight: bold;
        font-size: 3.5rem;
        color: #FFFFFF;
        text-align: right;
        line-height: 1.4;
        width: 100%;
        margin: 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        padding-bottom: 2rem;
    }}
    
    /* Botones mejorados */
    .stButton>button {{
        min-width: 200px;
        height: 3.2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 12px;
        background-color: #FFFFFF;
        color: black;
        border: none;
        transition: all 1.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }}
    
    .stButton>button:hover {{
        background-color: #43a047;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }}
    
    /* Media queries para responsividad */
    @media (max-width: 1200px) {{
        .overlay-container {{
            max-width: 90%;
            padding: 3rem 2rem;
        }}
        .title {{
            font-size: 2.2rem;
        }}
    }}
    
    @media (max-width: 768px) {{
        .overlay-container {{
            max-width: 95%;
            padding: 2.5rem 1.5rem;
            margin: 5vh auto;
            min-height: 450px;
        }}
        .title {{
            font-size: 1.8rem;
            text-align: center;
            padding-bottom: 1.5rem;
        }}
        .button-columns {{
            flex-direction: column;
        }}
        .stButton>button {{
            min-width: 250px;
            width: 100%;
        }}
    }}
    
    @media (max-width: 480px) {{
        .overlay-container {{
            padding: 2rem 1rem;
        }}
        .title {{
            font-size: 1.6rem;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)

    
    # TÍTULO DENTRO DEL OVERLAY
    st.markdown("""
    <div class='title'>
    Análisis de datos y modelado predictivo<br>
    sobre la accidentabilidad vial<br>
    en el Valle de Aburrá<br>
    (2015–2019)
    </div>
    """, unsafe_allow_html=True)
    
    # BOTONES DENTRO DEL OVERLAY
    st.markdown("<div class='button-columns'>", unsafe_allow_html=True)
    
    # Crear columnas para los botones
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        if st.button("📊 Análisis Exploratorio", use_container_width=True):
            st.session_state["pagina"] = "analisis"
            st.rerun()
    with col2:
        if st.button("🤖 Modelo Predictivo", use_container_width=True):
            st.session_state["pagina"] = "modelo"
            st.rerun()
    with col3:
        if st.button("📈 Curvas y Resultados", use_container_width=True):
            st.session_state["pagina"] = "curvas"
            st.rerun()
    with col4:
        if st.button("🧾 Resumen Ejecutivo", use_container_width=True):
            st.session_state["pagina"] = "resumen"
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)  # Cierra button-columns
    st.markdown("</div>", unsafe_allow_html=True)  # Cierra content-wrapper
    st.markdown("</div>", unsafe_allow_html=True)  # Cierra overlay-container

# ======================================================
# FUNCIÓN DASHBOARD PRINCIPAL (MANTENIDA)
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

    # Aplicar estilo consistente al dashboard
    st.markdown("""
    <style>
    .dashboard-container {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🚦 Proyecto Accidentabilidad Vial Antioquia — Panel General")

    # Botón volver a portada mejorado
    col_volver, _ = st.columns([1, 5])
    with col_volver:
        if st.button("🏠 Volver a la Portada", use_container_width=True):
            st.session_state["pagina"] = "inicio"
            st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Análisis Exploratorio",
        "🤖 Modelo Predictivo", 
        "📈 Curvas y Matrices",
        "🧾 Resumen Ejecutivo"
    ])

    with tab1:
        st.markdown("<div class='dashboard-container'>", unsafe_allow_html=True)
        st.header("Distribución de Accidentes por Variables Clave")
        col1, col2 = st.columns(2)
        with col1:
            st.image(imagenes["gravedad"], use_column_width=True)
            st.image(imagenes["jornada"], use_column_width=True)
        with col2:
            st.image(imagenes["clase"], use_column_width=True)
            st.image(imagenes["comuna"], use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='dashboard-container'>", unsafe_allow_html=True)
        st.header("Importancia de Variables — Modelo Random Forest")
        st.dataframe(datos["importancia"], use_container_width=True)
        st.subheader("Predicciones Simuladas")
        st.dataframe(datos["predicciones"], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='dashboard-container'>", unsafe_allow_html=True)
        st.header("Evaluación del Modelo de Clasificación")
        col3, col4 = st.columns(2)
        with col3:
            st.image(imagenes["roc"], use_column_width=True)
        with col4:
            st.image(imagenes["matriz"], use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        st.markdown("<div class='dashboard-container'>", unsafe_allow_html=True)
        st.header("📄 Resumen Ejecutivo del Proyecto")
        st.text(datos["resumen"])
        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# CONTROL DE NAVEGACIÓN
# ======================================================

if "pagina" not in st.session_state:
    st.session_state["pagina"] = "inicio"

if st.session_state["pagina"] == "inicio":
    mostrar_portada()
else:
    mostrar_dashboard()
