import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv
from huggingface_hub import login

# =========================
# FORCE HF CACHE (D DRIVE)
# =========================
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["HF_HUB_CACHE"] = "D:/hf_cache"

# =========================
# LOAD ENV & LOGIN
# =========================
load_dotenv()
login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

# =========================
# CONFIG
# =========================
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "models/tinyllama-lora"

# =========================
# ROBOTICS DOMAIN KEYWORDS
# =========================
ROBOTICS_KEYWORDS = [
    "robot", "robotics",
    "slam", "localization", "mapping",
    "ros", "ros2", "navigation",
    "sensor", "lidar", "laser", "imu",
    "control", "pid",
    "path", "planning",
    "actuator", "motor",
    "autonomous", "automation"
]


# =========================
# STREAMLIT PAGE CONFIG
# =========================
st.set_page_config(
    page_title="🤖 TinyLlama Robotics Chatbot",
    page_icon="🤖",
    layout="centered"
)

# =========================
# CUSTOM CSS (PYARA UI)
# =========================
st.markdown("""
<style>
.user-bubble {
    background: #2563eb;
    color: white;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 10px;
    max-width: 80%;
}
.bot-bubble {
    background: #1e293b;
    color: #e5e7eb;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 15px;
    max-width: 80%;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL (CACHED)
# =========================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="cpu"
    )
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Settings")
temperature = st.sidebar.slider("Creativity", 0.3, 1.2, 0.7)
max_tokens = st.sidebar.slider("Max response length", 50, 300, 150)

st.sidebar.markdown("---")
st.sidebar.markdown("🧠 **Model:** TinyLlama + LoRA")
st.sidebar.markdown("📚 **Domain:** Robotics ONLY")
st.sidebar.markdown("💻 **Mode:** CPU")

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# TITLE
# =========================
st.title("🤖 TinyLlama Robotics Chatbot")
st.caption("LoRA-fine-tuned • Anti-hallucination • Domain-restricted")

# =========================
# DISPLAY CHAT
# =========================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='user-bubble'>🧑 {msg['content']}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='bot-bubble'>🤖 {msg['content']}</div>",
            unsafe_allow_html=True
        )

# =========================
# USER INPUT
# =========================
user_input = st.chat_input("Ask something about robotics...")

if user_input is not None and user_input.strip() != "":

    user_text = user_input.lower()

    # HARD BLOCK FOR RRT / CODE
    rrt_keywords = ["rrt", "rrt*", "path planning"]
    if any(k in user_text for k in rrt_keywords):
        safe_msg = (
            "⚠️ **Conceptual explanation only**\n\n"
            "RRT and similar path-planning algorithms are explained conceptually "
            "to avoid unverified implementations."
        )
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": safe_msg})
        st.rerun()

    # DOMAIN CHECK
    if not any(word in user_text for word in ROBOTICS_KEYWORDS):
        refusal_msg = (
            "❌ **Out of domain**\n\n"
            "This assistant answers only robotics-related questions."
        )
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": refusal_msg})
        st.rerun()

    # =========================
    # SAVE USER MESSAGE
    # =========================
    st.session_state.messages.append({"role": "user", "content": user_input})

    # =========================
    # STRICT SYSTEM PROMPT
    # =========================
    prompt = """<|system|>
You are a STRICT domain-specific assistant.

Your domain is ONLY robotics, automation, and intelligent machines.

Rules:
- Never guess or hallucinate.
- Never invent libraries or APIs.
- If unsure, explain conceptually.
- Prefer correctness over completeness.
-Avoid making assumptions not stated in the question.

<|assistant|>
"""

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            prompt += f"<|user|>\n{msg['content']}\n"
        else:
            prompt += f"<|assistant|>\n{msg['content']}\n"

    prompt += "<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt")

    with st.spinner("🤖 Thinking..."):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id
            )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = response.split("<|assistant|>")[-1].strip()

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
