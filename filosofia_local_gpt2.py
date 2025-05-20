import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Configuración de la app
st.set_page_config(page_title="Generador Filosófico en Español", page_icon="📚")
st.title("📚 Generador de Texto Filosófico (Español, local)")

# Carga del modelo en español (desde Hugging Face)
@st.cache_resource
def cargar_modelo():
    modelo = GPT2LMHeadModel.from_pretrained("datificate/gpt2-small-spanish")
    tokenizer = GPT2Tokenizer.from_pretrained("datificate/gpt2-small-spanish")
    return modelo, tokenizer

modelo, tokenizer = cargar_modelo()

# Entrada del usuario
tema = st.text_input("📝 Tema filosófico", placeholder="Ej: El tiempo y la existencia")
longitud = st.slider("📏 Longitud del texto", 50, 300, 150)

# Selector de estilo
estilo = st.selectbox("🧠 Estilo filosófico", ["General", "Socrático", "Existencialista", "Estoico", "Idealismo alemán"])

# Construcción del prompt inicial
def construir_prompt(tema, estilo):
    estilos = {
        "General": f"Reflexionando sobre {tema}, uno puede considerar que",
        "Socrático": f"– Sócrates: ¿Qué es {tema}?\n– Interlocutor:",
        "Existencialista": f"En un mundo sin sentido, {tema} se convierte en reflejo de la angustia del ser.",
        "Estoico": f"Un sabio estoico contempla {tema} con serenidad y razón.",
        "Idealismo alemán": f"Según la razón pura, {tema} es una manifestación del espíritu absoluto.",
    }
    return estilos.get(estilo, f"Reflexión sobre {tema}:")

# Botón para generar
if st.button("Generar reflexión"):
    if not tema:
        st.warning("Por favor, escribe un tema.")
    else:
        prompt = construir_prompt(tema, estilo)

        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Generar texto
        with torch.no_grad():
            salida = modelo.generate(
                input_ids,
                max_length=len(input_ids[0]) + longitud,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                temperature=0.95,
                top_k=50,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        texto_generado = tokenizer.decode(salida[0], skip_special_tokens=True)
        st.markdown("### 🧾 Reflexión generada")
        st.write(texto_generado)
