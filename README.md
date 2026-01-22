#  Robotics Domain Chatbot (LoRA-Fine-Tuned)

This repository contains a **domain-specific chatbot** fine-tuned to answer questions about **Robotics**, using a  **LoRA-fine-tuned TinyLlama model**.

Unlike generic AI chatbots that hallucinate answers, this assistant is:
- 🧠 **Conservative and Trustworthy** — prioritizes factual correctness over confident output
- 📘 **Dataset-Aligned** — limited to robotics domain knowledge
- ⚙️ **Lightweight & Efficient** — LoRA fine-tuned on a small model
- 🌐 **Deployable** on Streamlit Community Cloud

---

## 🧠 Project Overview

Robotics is a technical field with precise terminology and nuanced concepts like:
- Localization (AMCL)
- SLAM (Simultaneous Localization and Mapping)
- ROS vs ROS2
- Sensor fusion and control systems

This chatbot is designed to:
- Provide **high-quality, robotics-specific answers**
- Reject out-of-domain requests
- Avoid hallucination using prompt engineering and guardrails
- Provide conservative responses comparable to interview expectations

The project combines:
- **LoRA fine-tuning** to adapt a small LLM to your dataset  
- **Prompt-level anti-hallucination logic**  
- **Streamlit UI** for interactive chats  
- Domain filtering logic to enforce relevance

📌 *This is not a generic chatbot — it is a domain-restricted, interview-ready assistant.*

---

## 🧩 How It Works

### 1. **Dataset Preparation**
Your dataset (`dataset.jsonl`) contains structured Q&A pairs or text related to robotics topics. This is used during fine-tuning to teach the base model specializations.

---

### 2. **LoRA Fine-Tuning**
LoRA (Low-Rank Adaptation) allows fine-tuning of large models with limited resources by training only small adapter matrices.

The `training.py` script:
- Loads base model & tokenizer
- Prepares dataset
- Applies LoRA configuration
- Trains for a fixed number of steps
- Saves the fine-tuned weights

---

### 3. **Inference**
The `inference.py` script:
- Loads the LoRA-adapted model
- Uses safe prompting
- Generates responses using the TinyLlama base

---

### 4. **Interactive UI**
`app.py`:
- Front-end UI powered by **Streamlit**
- Takes user questions
- Applies domain filters
- Generates safe and reliable answers

---

## 🚀 Deployment on Streamlit

To deploy your chatbot online (Streamlit Community Cloud):

1. Push your code to a **GitHub repository**
2. Go to **https://share.streamlit.io**
3. Connect your GitHub repo
4. Select the `main` branch and `app.py`
5. Add your Hugging Face token as a **secret**
   ```toml
   HUGGINGFACE_HUB_TOKEN = "<your_hf_token>"
