# ======================================================
# DASHBOARD STREAMLIT — Proyecto Accidentabilidad Vial Antioquia
# ======================================================

import streamlit as st
from pathlib import Path
from PIL import Image
import pandas as pd
import base64
from streamlit_elements import elements, mui, html, dashboard

# ======================================================
# CONFIGURACIÓN INICIAL
# ======================================================
st.set_page_config(page_title="Dashboard — Accidentabilidad Vial Valle de Aburrá",
                   layout="wide", page_icon="🚦")

BASE_DIR = Path(__file__).parent.resolve()
GRAF_DIR = BASE_DIR / "Graficas_Salida"
MODEL_DIR = BASE_DIR / "Modelo_Predict"
IMG_PATH = BASE_DIR / "Diseño" / "Map_portada.png"

# ======================================================
# FUNCIÓN UTIL
# ======================================================
def img_to_base64(path: Path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

def pil_to_base64_img_tag(pil_img: Image.Image, fmt="PNG"):
    from io import BytesIO
    buf = BytesIO()
    pil_img.save(buf, format=fmt)
    data = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{data}"

# ======================================================
# SESIONES INICIALES
# ======================================================
if "vista" not in st.session_state:
    st.session_state["vista"] = "portada"

if "titulo_portada" not in st.session_state:
    st.session_state["titulo_portada"] = (
        "Análisis de datos y modelado predictivo\n"
        "sobre la accidentalidad vial\n"
        "en el Valle de Aburrá\n"
        "(2015–2019)"
    )

if "fondo_portada" not in st.session_state:
    st.session_state["fondo_portada"] = img_to_base64(IMG_PATH)

if "transparencia_portada" not in st.session_state:
    st.session_state["transparencia_portada"] = 0.88

def go(v):
    st.session_state["vista"] = v

# ======================================================
# PORTADA EDITABLE
# ======================================================
def mostrar_portada():
    bg_data = st.session_state["fondo_portada"]

    st.markdown(f"""
        <style>
        .portada {{
            width: 100%;
            height: 78vh;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-image: url("data:image/png;base64,{bg_data}");
            background-position: center;
            background-size: cover;
            background-repeat: no-repeat;
            padding: 3rem;
            box-sizing: border-box;
        }}
        .portada_box {{
            background: rgba(255,255,255,{st.session_state["transparencia_portada"]});
            padding: 2rem 2.8rem;
            border-radius: 14px;
            max-width: 55%;
            margin-left: auto;
            margin-right: 4%;
            box-shadow: 0 8px 30px rgba(0,0,0,0.18);
            display:flex;
            flex-direction:column;
            justify-content:center;
        }}
        .titulo {{
            font-family: "Times New Roman", serif;
            font-weight: 700;
            font-size: 34px;
            color: #0f1720;
            text-align: right;
            line-height: 1.35;
            white-space: pre-line;
        }}
        .stButton>button {{
            min-width:160px;
            height:44px;
            border-radius:10px;
            background-color:#2e7d32;
            color:white;
            font-weight:600;
        }}
        .stButton>button:hover {{ background-color:#43a047; }}
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='portada'>", unsafe_allow_html=True)
    st.markdown("<div style='width:28%;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='portada_box'>", unsafe_allow_html=True)

    st.markdown(f"<div class='titulo'>{st.session_state['titulo_portada']}</div>", unsafe_allow_html=True)

    # Botones navegación
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("⚙️ Editar Portada"):
            go("editar_portada")
    with col2:
        if st.button("📊 Análisis Exploratorio"):
            go("dashboard")
    with col3:
        if st.button("🤖 Modelo Predictivo"):
            go("modelo")
    with col4:
        if st.button("📈 Curvas y Resultados"):
            go("curvas")
    with col5:
        if st.button("🧾 Resumen Ejecutivo"):
            go("resumen")

    st.markdown("</div></div>", unsafe_allow_html=True)

# ======================================================
# MODO EDICIÓN DE PORTADA
# ======================================================
def editar_portada():
    st.subheader("⚙️ Configurar Portada del Dashboard")

    st.text_area("Título principal", st.session_state["titulo_portada"], key="titulo_portada")

    uploaded = st.file_uploader("Sube una nueva imagen de fondo (PNG o JPG)", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Previsualización", use_column_width=True)
        st.session_state["fondo_portada"] = pil_to_base64_img_tag(img).split(",")[1]

    st.slider("Nivel de transparencia (0 opaco — 1 invisible)",
              0.5, 1.0, st.session_state["transparencia_portada"], key="transparencia_portada")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Guardar Cambios"):
            st.success("Cambios guardados.")
            go("portada")
    with c2:
        if st.button("❌ Cancelar"):
            go("portada")

# ======================================================
# VISTAS PLACEHOLDER (ya integradas)
# ======================================================
def mostrar_dashboard():
    st.header("📊 Análisis Exploratorio — Dashboard Interactivo (en construcción)")
    if st.button("🏠 Volver a Portada"):
        go("portada")

def mostrar_modelo():
    st.header("🤖 Modelo Predictivo — Resultados")
    if st.button("🏠 Volver a Portada"):
        go("portada")

def mostrar_curvas():
    st.header("📈 Curvas y Resultados del Modelo")
    if st.button("🏠 Volver a Portada"):
        go("portada")

def mostrar_resumen():
    st.header("🧾 Resumen Ejecutivo del Proyecto")
    if st.button("🏠 Volver a Portada"):
        go("portada")

# ======================================================
# CONTROLADOR PRINCIPAL
# ======================================================
def main():
    vista = st.session_state.get("vista", "portada")

    if vista == "portada":
        mostrar_portada()
    elif vista == "editar_portada":
        editar_portada()
    elif vista == "dashboard":
        mostrar_dashboard()
    elif vista == "modelo":
        mostrar_modelo()
    elif vista == "curvas":
        mostrar_curvas()
    elif vista == "resumen":
        mostrar_resumen()
    else:
        mostrar_portada()

if __name__ == "__main__":
    main()
