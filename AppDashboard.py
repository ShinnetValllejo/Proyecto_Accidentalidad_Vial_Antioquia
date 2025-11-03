# ======================================================
# DASHBOARD STREAMLIT — Proyecto Accidentabilidad Vial Antioquia
# ======================================================

import streamlit as st
from pathlib import Path
from PIL import Image
import pandas as pd
import os
from pathlib import Path

# ======================================================
# CONFIGURACIÓN INICIAL
# ======================================================
st.set_page_config(
    page_title="Proyecto Accidentabilidad Vial Antioquia",
    layout="wide",
    page_icon="🚦"
)

# ======================================================
# RUTAS PRINCIPALES (IGUALES AL SCRIPT DE MODELO)
# ======================================================
BASE_DIR = Path(__file__).parent.resolve()
GRAF_DIR = BASE_DIR / "Graficas_Salida"
MODEL_DIR = BASE_DIR / "Modelo_Predict"

# ======================================================
# CARGA DE ARCHIVOS — CACHEADA PARA EVITAR REDIBUJOS
# ======================================================
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

# ======================================================
# CARGA INICIAL
# ======================================================
imagenes = load_images()
datos = load_data()

# ======================================================
# TÍTULO
# ======================================================
st.title("🚧 Proyecto Accidentabilidad Vial Antioquia — Dashboard General")

# ======================================================
# PESTAÑAS PRINCIPALES
# ======================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Análisis Exploratorio",
    "🤖 Modelo Predictivo",
    "📈 Curvas y Matrices",
    "🧾 Resumen Ejecutivo"
])

# ======================================================
# TAB 1 — ANÁLISIS EXPLORATORIO
# ======================================================
with tab1:
    st.header("Distribución de Accidentes por Variables Clave")

    col1, col2 = st.columns(2)
    with col1:
        st.image(imagenes["gravedad"], caption="Distribución por Gravedad de Accidentes")
        st.image(imagenes["jornada"], caption="Accidentes por Jornada del Día")
    with col2:
        st.image(imagenes["clase"], caption="Tipos de Accidente más Frecuentes")
        st.image(imagenes["comuna"], caption="Top 10 Comunas con Mayor Número de Accidentes")

    st.markdown("""
    **Observaciones principales:**
    - Alta concentración de accidentes catalogados como “Solo Daños”.
    - Mayor frecuencia en comunas urbanas con tráfico denso.
    - Jornadas de la tarde presentan los picos de mayor ocurrencia.
    """)

# ======================================================
# TAB 2 — MODELO PREDICTIVO
# ======================================================
with tab2:
    st.header("Importancia de Variables — Modelo Random Forest")
    st.dataframe(datos["importancia"], use_container_width=True)

    st.markdown("""
    **Interpretación:**  
    Las variables con mayor importancia determinan el impacto de cada característica en la predicción de la gravedad del accidente.
    """)

    st.divider()
    st.subheader("Predicciones Simuladas (Nuevos Casos)")
    st.dataframe(datos["predicciones"], use_container_width=True)

# ======================================================
# TAB 3 — CURVAS Y MATRICES
# ======================================================
with tab3:
    st.header("Evaluación del Modelo de Clasificación")

    col3, col4 = st.columns(2)
    with col3:
        st.image(imagenes["roc"], caption="Curva ROC — Evaluación del Modelo")
        st.markdown("""
        **Curva ROC:**  
        Muestra la capacidad del modelo para distinguir entre clases.  
        Un AUC alto indica buen rendimiento predictivo.
        """)
    with col4:
        st.image(imagenes["matriz"], caption="Matriz de Confusión — Random Forest")
        st.markdown("""
        **Matriz de Confusión:**  
        Permite evaluar qué tan bien el modelo clasifica los casos de “Solo Daños” y “Con Heridos”.
        """)

# ======================================================
# TAB 4 — RESUMEN EJECUTIVO
# ======================================================
with tab4:
    st.header("📄 Resumen Ejecutivo del Proyecto")
    st.text(datos["resumen"])

    st.markdown("""
    **Conclusión General:**  
    El modelo Random Forest implementado ofrece una predicción confiable sobre la gravedad de accidentes en Antioquia.  
    El análisis exploratorio y los hallazgos clave apoyan la toma de decisiones en seguridad vial y prevención de siniestros.
    """)

# ======================================================
# PIE DE PÁGINA
# ======================================================
st.markdown("---")
st.markdown("**Desarrollado por:** Equipo de Análisis y Calidad de Datos — Proyecto SVA 🚦")