import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from config import check_api_keys
from agent_logic import prompt_ai

# --- Check API keys ---
check_api_keys()

# --- Streamlit UI ---
st.set_page_config(page_title="EuclidIA | Think. Explain. Prove.", page_icon="📐")

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 📐 EuclidIA")
    st.markdown("*Think. Explain. Prove.*")
    st.markdown("*An AI-powered assistant for mathematics*")
    st.markdown("---")
    st.markdown("**Two powerful modes, one goal**")
    st.markdown("💡 *Explain with clarity*")
    st.markdown("🧠 *Reason with precision*")
    st.markdown("---")
    st.markdown("**Contact**")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/adel-messaoudi-831358132)")
    st.markdown("[GitHub](https://github.com/AdelMessaoudi-13)")
    st.markdown("[Hugging Face](https://huggingface.co/AdelMessaoudi-13)")

# --- Header (centered, professional) ---
st.markdown("""
<style>
    .header-container {
        max-width: 800px;
        margin-left: 0;
        padding: 1rem 1rem 0.5rem 1rem;
        font-family: "Segoe UI", sans-serif;
    }
</style>

<div class="header-container">
    <h1>📐 EuclidIA</h1>
</div>
""", unsafe_allow_html=True)

# --- Conversation context ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="""
# 🎯 Role
You are an AI assistant specialized in mathematics. You must answer only questions related to mathematics.

# 🛠️ Available Tools
You have access to two tools to answer questions:

- `use_gemini`: for definitions, clear explanations of mathematical concepts, established properties, formulas, or any factual response.
- `use_deepseek`: for proofs, formal demonstrations, detailed reasoning, or problem solving that requires multiple logical steps.

# 🧭 Guidelines
Carefully analyze each question and choose the most appropriate tool:
- If the question is straightforward, factual, or asks for a simple explanation → use `use_gemini`.
- If the question requires structured reasoning, rigorous justification, or a demonstration → use `use_deepseek`.

Always use **only one** of these two tools to answer.
""")
    ]

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Initialize model tracking
#if "current_model" not in st.session_state:
#    st.session_state.current_model = None

# --- Input field and buttons ---
col1, col2, col3 = st.columns([6, 0.7, 0.7])
with col1:
    user_input = st.text_input(
        label="Math question input",
        placeholder="Ask your question...",
        label_visibility="collapsed",
        key="input_field"  # no value set here
    )
with col2:
    send_clicked = st.button("➤", help="Send", use_container_width=True)
with col3:
    clear_clicked = st.button("🗑️", help="Clear", use_container_width=True)

# --- Clear logic ---
if clear_clicked:
    st.session_state.user_input = ""
    st.session_state.messages = st.session_state.messages[:1]
    st.session_state.pop("input_field", None)  # <- vide proprement
    st.rerun()

# --- Auto-send on Enter ---
if "input_field" in st.session_state:
    input_value = st.session_state["input_field"].strip()
else:
    input_value = ""

input_changed = input_value != "" and input_value != st.session_state.user_input

if input_changed and not send_clicked:
    send_clicked = True

# --- Processing ---
if send_clicked and input_value:
    st.session_state.user_input = input_value

    st.session_state.loading_placeholder = st.empty()
    st.session_state.loading_placeholder.markdown("⏳ **Thinking...**")

    try:
        st.session_state.messages.append(HumanMessage(content=input_value))
        response = prompt_ai(st.session_state.messages)
        st.session_state.messages.append(response)

        st.session_state.loading_placeholder.empty()

        if hasattr(response, "content") and response.content:
            st.success("Assistant's response:")
            st.markdown(response.content, unsafe_allow_html=True)
        else:
            st.warning("No response was generated.")
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        if 'loading_placeholder' in st.session_state:
            del st.session_state.loading_placeholder
