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

    # Log the input sent to Gemini API
    #print(f"\n[USE_GEMINI] 🔍 Received question:\n{question}")

    try:
        response = llm_gemini.invoke(question)

        # Log the response content from Gemini API
        #print(f"[USE_GEMINI] ✅ API response:\n{response.content}")

        return response.content
    except Exception as e:
        # Log the exception from Gemini API
        #print(f"[USE_GEMINI] ❌ API failed:\n{e}")
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

    # Log the input sent to DeepSeek API
    #print(f"\n[USE_DEEPSEEK] 🔍 Received question:\n{question}")

    try:
        response = llm_deepseek.invoke(question)

        # Log the response content from DeepSeek API
        #print(f"[USE_DEEPSEEK] ✅ API response:\n{response.content}")

        return response.content
    except Exception as e:
        # Log the exception from DeepSeek API
        #print(f"[USE_DEEPSEEK] ❌ API failed:\n{e}")
        st.error(f"DeepSeek failed: {e}")
        return f"[ERROR] DeepSeek failed: {e}"
    finally:
        # Always clear the loading placeholder after response or error
        if 'loading_placeholder' in st.session_state:
            st.session_state.loading_placeholder.empty()
