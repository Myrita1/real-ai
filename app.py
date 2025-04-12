import streamlit as st
import torch
from tokenizer import encode, decode
from model import AceAssistantModel

# Set Streamlit page config
st.set_page_config(page_title="Ace Assistant LLM", layout="centered")
st.title("🤖 Ace Assistant Language Model")
st.markdown("Welcome! Enter a prompt and see how the model responds.")

# Load model
@st.cache_resource
def load_model():
    model = AceAssistantModel()
    model.load_state_dict(torch.load("ace_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# User input
prompt = st.text_area("🗣️ Your Prompt", height=150)

# Generate response
if st.button("Generate Response"):
    if not prompt.strip():
        st.warning("Please enter a prompt to get a response.")
    else:
        try:
            # Encode input
            input_ids = torch.tensor([encode(prompt)])
            with torch.no_grad():
                output = model(input_ids)
                predicted_ids = torch.argmax(output, dim=-1)[0]
                response = decode(predicted_ids.tolist())

            st.subheader("📝 Model Response")
            st.write(response)

        except Exception as e:
            st.error(f"An error occurred: {e}")
