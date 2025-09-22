import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained('./model')
tokenizer = AutoTokenizer.from_pretrained('./model')

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit app
st.title("Robotics Chatbot")

user_input = st.text_input("Ask me anything about robotics:")

if user_input:
    prompt = f"Question: {user_input}\nAnswer:"
    response = generate_response(prompt)
    st.write(response)
