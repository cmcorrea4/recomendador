"""
Recomendador Semántico — Streamlit
Busca, por similitud de significado (embeddings), la respuesta más cercana
dentro de un diccionario de pregunta→respuesta predefinido.
"""

import json

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Recomendador Semántico", page_icon="🔎", layout="centered")


# ── Modelo (se carga una sola vez y se reutiliza entre recargas) ────────────
@st.cache_resource(show_spinner="Cargando modelo de embeddings...")
def cargar_modelo():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


modelo = cargar_modelo()

# ── Diccionario de referencia por defecto (pregunta → respuesta) ───────────
DICCIONARIO_DEFECTO = {
    "¿Qué es un embedding?": (
        "Un embedding es una representación numérica densa de un texto que captura "
        "su significado, ubicando frases similares cerca en un espacio vectorial."
    ),
    "¿Qué es el one-hot encoding?": (
        "Es una representación donde cada palabra es un vector con un único 1 en la "
        "posición correspondiente del vocabulario y 0 en el resto; no captura significado."
    ),
    "¿Qué es la similitud del coseno?": (
        "Es una medida que compara el ángulo entre dos vectores para determinar qué tan "
        "parecidos son en dirección (significado), sin importar su magnitud."
    ),
    "¿Qué es la tokenización?": (
        "Es el proceso de dividir un texto en unidades más pequeñas, como oraciones o "
        "palabras, para poder analizarlas."
    ),
    "¿Qué son las stopwords?": (
        "Son palabras muy frecuentes (como 'el', 'de', 'que') que aportan poco significado "
        "propio y suelen filtrarse antes del análisis."
    ),
    "¿Qué es PCA?": (
        "Es una técnica de reducción de dimensionalidad que proyecta vectores de muchas "
        "dimensiones a 2 o 3 dimensiones para poder visualizarlos."
    ),
    "¿Qué es la Ley de Zipf?": (
        "Es un patrón estadístico del lenguaje donde pocas palabras concentran la mayoría "
        "de las apariciones en un texto."
    ),
    "¿Qué modelo se usa para generar embeddings en este proyecto?": (
        "Se usa sentence-transformers con un modelo multilingüe que corre localmente, "
        "sin necesidad de una API de pago."
    ),
}

if "diccionario" not in st.session_state:
    st.session_state.diccionario = DICCIONARIO_DEFECTO.copy()


def similitud_coseno(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@st.cache_data(show_spinner="Calculando embeddings del diccionario...")
def calcular_embeddings(preguntas: tuple):
    return modelo.encode(list(preguntas))


# ── Barra lateral: gestión del diccionario ─────────────────────────────────
with st.sidebar:
    st.header("📚 Diccionario de referencia")

    modo = st.radio(
        "Fuente del diccionario",
        ["Usar el predeterminado", "Cargar un JSON propio", "Editar manualmente"],
    )

    if modo == "Usar el predeterminado":
        st.session_state.diccionario = DICCIONARIO_DEFECTO.copy()

    elif modo == "Cargar un JSON propio":
        st.caption('Formato esperado: `{"pregunta": "respuesta", ...}`')
        archivo = st.file_uploader("Sube tu archivo .json", type="json")
        if archivo is not None:
            try:
                nuevo = json.load(archivo)
                if not isinstance(nuevo, dict) or len(nuevo) == 0:
                    st.error("El JSON debe ser un objeto no vacío de pregunta → respuesta.")
                else:
                    st.session_state.diccionario = nuevo
                    st.success(f"✅ {len(nuevo)} pares cargados")
            except Exception as e:
                st.error(f"No se pudo leer el archivo: {e}")

    elif modo == "Editar manualmente":
        texto = st.text_area(
            "Edita el JSON directamente",
            value=json.dumps(st.session_state.diccionario, ensure_ascii=False, indent=2),
            height=320,
        )
        if st.button("Aplicar cambios"):
            try:
                nuevo = json.loads(texto)
                if not isinstance(nuevo, dict) or len(nuevo) == 0:
                    st.error("El JSON debe ser un objeto no vacío de pregunta → respuesta.")
                else:
                    st.session_state.diccionario = nuevo
                    st.success("✅ Diccionario actualizado")
            except Exception as e:
                st.error(f"JSON inválido: {e}")

    st.divider()
    umbral = st.slider("Umbral mínimo de confianza", 0.0, 1.0, 0.35, 0.05)
    top_k = st.slider("Alternativas a mostrar", 1, 5, 3)

    st.divider()
    st.caption(f"📌 {len(st.session_state.diccionario)} pares en el diccionario actual")


# ── Cuerpo principal ────────────────────────────────────────────────────────
st.title("🔎 Recomendador Semántico")
st.caption(
    "Escribe una pregunta y el sistema buscará, por **significado** (no por texto exacto), "
    "la más parecida dentro del diccionario de referencia."
)

diccionario = st.session_state.diccionario
preguntas_ref = list(diccionario.keys())
respuestas_ref = list(diccionario.values())

if len(preguntas_ref) == 0:
    st.warning("El diccionario está vacío. Carga uno desde la barra lateral.")
    st.stop()

embeddings_ref = calcular_embeddings(tuple(preguntas_ref))

pregunta_usuario = st.text_input(
    "✍️ Escribe tu pregunta:",
    placeholder="Ej: ¿para qué sirve reducir la dimensión de un vector?",
)

if pregunta_usuario:
    emb_usuario = modelo.encode(pregunta_usuario)
    similitudes = [similitud_coseno(emb_usuario, e) for e in embeddings_ref]
    orden = np.argsort(similitudes)[::-1]

    mejor_idx = orden[0]
    mejor_sim = similitudes[mejor_idx]

    st.divider()

    if mejor_sim < umbral:
        st.warning(
            f"No encontré una coincidencia suficientemente segura "
            f"(similitud máxima: {mejor_sim:.2%}). Intenta reformular la pregunta "
            f"o baja el umbral en la barra lateral."
        )
    else:
        st.subheader("✅ Respuesta")
        st.write(respuestas_ref[mejor_idx])
        st.caption(f"Coincide con: _\"{preguntas_ref[mejor_idx]}\"_ — confianza: {mejor_sim:.2%}")

    with st.expander(f"Ver las {top_k} coincidencias más cercanas"):
        for idx in orden[:top_k]:
            st.markdown(f"**{similitudes[idx]:.2%}** — {preguntas_ref[idx]}")
            st.caption(respuestas_ref[idx])
            st.write("")
else:
    st.info("Escribe una pregunta arriba para buscar en el diccionario.")
