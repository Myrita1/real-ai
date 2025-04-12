import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"  # Prevent torch.classes crash in Streamlit

import streamlit as st
import torch
from model import AceAssistantModel
from tokenizer import encode_text, decode_output  # assumes you have encode/decode helpers

# App title
st.set_page_config(page_title="Ace Assistant LLM", layout="centered")
st.title("🤖 Ace Assistant LLM")
st.markdown("Enter a prompt below and let your custom LLM respond!")

# Initialize model (same shape as training config)
@st.cache_resource
def load_model():
    model = AceAssistantModel()
    model.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# User input
prompt = st.text_area("📝 Prompt", height=200)

# Inference
if st.button("Generate Response"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        try:
            input_ids = encode_text(prompt)  # convert to tensor
            with torch.no_grad():
                output_logits = model(input_ids)
                predicted_ids = torch.argmax(output_logits, dim=-1)
                response = decode_output(predicted_ids)
            st.success("✅ Response:")
            st.write(response)
        except Exception as e:
            st.error(f"⚠️ Error during generation: {e}")
