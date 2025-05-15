from langchain_core.tools import tool
import streamlit as st
from config import llms_config

# --- Load LLM ---
llm_gemini, llm_deepseek = llms_config.get_llms()

# --- Tools ---
@tool
def use_gemini(question: str) -> str:
    """Uses Gemini for simple math questions."""
    if 'loading_placeholder' in st.session_state:
        st.session_state.loading_placeholder.markdown("📘 **Explaining...**")

    try:
        response = llm_gemini.invoke(question)

        return response.content
    except Exception as e:
        st.error(f"Gemini failed: {e}")
        return f"[ERROR] Gemini failed: {e}"
    finally:
        # Always clear the loading placeholder after response or error
        if 'loading_placeholder' in st.session_state:
            st.session_state.loading_placeholder.empty()

@tool
def use_deepseek(question: str) -> str:
    """Uses DeepSeek for mathematical reasoning and proofs."""
    if 'loading_placeholder' in st.session_state:
        st.session_state.loading_placeholder.markdown("🧠 **Reasoning...**")

    try:
        response = llm_deepseek.invoke(question)

        return response.content
    except Exception as e:
        st.error(f"DeepSeek failed: {e}")
        return f"[ERROR] DeepSeek failed: {e}"
    finally:
        # Always clear the loading placeholder after response or error
        if 'loading_placeholder' in st.session_state:
            st.session_state.loading_placeholder.empty()
