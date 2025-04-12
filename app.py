import streamlit as st
import torch
from model import AceAssistantModel
from tokenizer import tokenize_input, decode_output

st.set_page_config(page_title="Ace Assistant LLM", layout="centered")

st.title("🧠 Ace Assistant Language Model")
st.write("Interact with your lightweight transformer model.")

user_input = st.text_area("Enter your prompt:", height=150)

if st.button("Generate"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_ids = tokenize_input(user_input)
        model = AceAssistantModel()
        model.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))
        model.eval()
        with torch.no_grad():
            output = model(input_ids)
            predicted_ids = output.argmax(dim=-1)
        result = decode_output(predicted_ids)
        st.success("Generated Output:")
        st.write(result)