import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from config import check_api_keys
from agent_logic import prompt_ai
from langchain_core.messages import ToolMessage
from tools import use_gemini, use_deepseek
import json
import base64

# --- Check API keys ---
check_api_keys()

# --- Streamlit UI ---
st.set_page_config(page_title="EuclidIA | Think. Explain. Prove.", page_icon="📐")

# Function to load and encode the logo
def get_img_as_base64(file_path):
    with open(file_path, "rb") as image_file:
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
    #print(f"[DEBUG] User input captured: '{input_value}'")

    st.session_state.user_input = input_value

    st.session_state.loading_placeholder = st.empty()
    st.session_state.loading_placeholder.markdown("⏳ **Thinking...**")

    try:
        st.session_state.messages.append(HumanMessage(content=input_value))

        #print("[DEBUG] Messages sent to agent:")
        #for msg in st.session_state.messages:
        #    print(f"  - [{msg.type}] {msg.content}")

        response = prompt_ai(st.session_state.messages)
        st.session_state.messages.append(response)

        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                args = tool_call["args"]

                #print(f"[DEBUG] Received tool_call:\n{json.dumps(tool_call, indent=2)}")
                question = args.get("question", "") if args else ""

                selected_tool = {
                    "use_gemini": use_gemini,
                    "use_deepseek": use_deepseek,
                }.get(tool_name)

                if selected_tool:
                    try:
                        tool_output = selected_tool.invoke(question)
                    except Exception as e:
                        tool_output = f"❌ Tool '{tool_name}' failed: {str(e)}"
                        st.error(tool_output)

                    st.session_state.messages.append(
                        ToolMessage(content=tool_output, tool_call_id=tool_call["id"])
                    )

            final_response = prompt_ai(st.session_state.messages)
            st.session_state.messages.append(final_response)

            if hasattr(final_response, "content") and final_response.content:
                st.session_state.loading_placeholder.empty() #
                st.success("Assistant's response:")
                st.markdown(final_response.content, unsafe_allow_html=True)
            else:
                st.session_state.loading_placeholder.empty() #
                st.warning("No response was generated after tool call.")
        else:
            if hasattr(response, "content") and response.content:
                st.session_state.loading_placeholder.empty() #
                st.success("Assistant's response:")
                st.markdown(response.content, unsafe_allow_html=True)
            else:
                st.session_state.loading_placeholder.empty() #
                st.warning("No response was generated.")

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        if 'loading_placeholder' in st.session_state:
            del st.session_state.loading_placeholder
