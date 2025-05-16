import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from config import check_api_keys
from agent_logic import prompt_ai
from tools import use_gemini, use_deepseek
import base64
import os

# --- Check API keys ---
check_api_keys()

# --- Streamlit UI ---
st.set_page_config(page_title="EuclidIA | Think. Explain. Prove.", page_icon="📐")

# Function to load and encode the logo
def get_img_as_base64(file_name):
    # Get absolute path of the file relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(script_dir, file_name)

    with open(abs_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

logo_base64 = get_img_as_base64('assets/euclidia_logo.png')

# --- Sidebar ---
with st.sidebar:
    st.markdown(f"""
    <div style='text-align: center;'>
        <img src="data:image/png;base64,{logo_base64}" width="140"><br>
        <div style='font-style: italic; margin-top: 0.3rem; line-height: 1.3;'>Think. Explain. Prove.</div>
    </div>
    <hr style='margin-top: 2rem; margin-bottom: 2rem;'>
    <div style='text-align: left; padding-left: 0.5rem;'>
        <strong>AI-powered mathematics assistant</strong>
        <div style='margin-top: 0.5rem;'>💡 <em>Explain with clarity</em></div>
        <div style='margin-top: 0.5rem;'>🧠 <em>Reason with precision</em></div>
    </div>
    <hr style='margin-top: 2rem; margin-bottom: 2rem;'>
    <div style='text-align: left; padding-left: 0.5rem;'>
        <strong>Contact</strong>
        <div style='margin-top: 0.5rem;'><a href="https://www.linkedin.com/in/adel-messaoudi-831358132">LinkedIn</a></div>
        <div style='margin-top: 0.5rem;'><a href="https://github.com/AdelMessaoudi-13">GitHub</a></div>
        <div style='margin-top: 0.5rem;'><a href="https://huggingface.co/AdelMessaoudi-13">Hugging Face</a></div>
    </div>
    """, unsafe_allow_html=True)

# --- Header (centered, professional) ---
st.markdown(f"""
<div style='display: flex; flex-direction: column; align-items: center; padding-top: 1rem;'>
    <img src="data:image/png;base64,{logo_base64}" style='width: 220px; margin-bottom: 1rem;'>
    <div style='width: 100%; max-width: 500px;'>
""", unsafe_allow_html=True)

# --- Conversation context ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="""
# Role
You are an AI assistant specialized in mathematics. You must answer only questions related to mathematics.

# Available Tools
You have access to two tools to answer questions:

- `use_gemini`: for definitions, clear explanations of mathematical concepts, established properties, formulas, or any factual response.
- `use_deepseek`: for proofs, formal demonstrations, detailed reasoning, or problem solving that requires multiple logical steps.

# Guidelines
Carefully analyze each question and choose the most appropriate tool:
- If the question is straightforward, factual, or asks for a simple explanation → use `use_gemini`.
- If the question requires structured reasoning, rigorous justification, or a demonstration → use `use_deepseek`.

Always use **only one** of these two tools to answer.
""")
    ]

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# --- Input field and buttons ---
col1, col2, col3 = st.columns([6, 0.7, 0.7])
with col1:
    user_input = st.text_input(
        label="Math question input",
        placeholder="Ask your question...",
        label_visibility="collapsed",
        key="input_field"
    )
with col2:
    send_clicked = st.button("➤", help="Send", use_container_width=True)
with col3:
    clear_clicked = st.button("🗑️", help="Clear", use_container_width=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# --- Clear logic ---
if clear_clicked:
    st.session_state.user_input = ""
    st.session_state.messages = st.session_state.messages[:1]
    st.session_state.pop("input_field", None)
    st.rerun()

# --- Processing ---
input_value = user_input.strip()

if send_clicked and input_value:
    # Check anti-duplication: don't add twice the same HumanMessage
    if not (st.session_state.messages and isinstance(st.session_state.messages[-1], HumanMessage) and st.session_state.messages[-1].content == input_value):
        # Store the user's input in the session state (for consistency)
        st.session_state.user_input = input_value

        # Create a placeholder to show the "Thinking..." indicator while processing
        st.session_state.loading_placeholder = st.empty()
        st.session_state.loading_placeholder.markdown("⏳ **Thinking...**")

        try:
            st.session_state.messages.append(HumanMessage(content=input_value))

            # Call the AI agent logic (prompt_ai) with the updated conversation history
            # The agent will handle tool calls, execution, and final synthesis by itself
            response = prompt_ai(st.session_state.messages)

            # Append the final AI response to the conversation history
            st.session_state.messages.append(response)

            # Display the final AI response if it contains text
            if hasattr(response, "content") and response.content:
                st.session_state.loading_placeholder.empty()
                st.success("Assistant's response:")
                st.markdown(response.content, unsafe_allow_html=True)
            else:
                st.session_state.loading_placeholder.empty()
                st.warning("No response was generated.")

        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            # Cleanup the loading placeholder when done (success or error)
            if 'loading_placeholder' in st.session_state:
                del st.session_state.loading_placeholder
