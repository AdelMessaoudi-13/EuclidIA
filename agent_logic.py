from tools import use_gemini, use_deepseek
from config import llms_config
from langchain_core.messages import AIMessage, SystemMessage
import streamlit as st

# --- Load LLMs & bind tools once ---
llm_gemini, llm_deepseek = llms_config.get_llms()
tools = [use_gemini, use_deepseek]
agent = llm_gemini.bind_tools(tools)

def clean_latex_with_gemini(text: str) -> str:
    """Uses Gemini to fix unformatted LaTeX expressions."""

    if 'loading_placeholder' in st.session_state:
        st.session_state.loading_placeholder.markdown("✨ **Formatting ...**")

    try:
        # Vérifier si le texte est trop long pour le nettoyage
        if len(text) > 8000:  # Limite de sécurité
            st.warning("Response too long for LaTeX cleaning, returning original.")
            return text

        cleaning_prompt = f"""
Role:
You are a post-processing assistant. You are given a math explanation that may contain LaTeX expressions that are not properly formatted.

Guidelines:
- Ensure all mathematical expressions are correctly formatted in LaTeX.
- Use $...$ for inline expressions.
- Use $$...$$ for block expressions.
- Avoid using \\( ... \\) or \\[ ... \\] for LaTeX formatting.
- Do not change the structure, content, or meaning of the explanation.
- Do not comment, explain, or add anything — return only the corrected version.

Here is the original text:

{text}
"""
        response = llm_gemini.invoke(cleaning_prompt)

        # Vérifier que la réponse n'est pas vide ou tronquée
        if not response.content or len(response.content.strip()) < len(text) * 0.5:
            st.warning("LaTeX cleaning may have truncated the response, returning original.")
            return text

        return response.content.strip()

    except Exception as e:
        st.warning(f"LaTeX cleaning failed: {e}. Returning original text.")
        return text
    finally:
        if 'loading_placeholder' in st.session_state:
            st.session_state.loading_placeholder.empty()

# --- Agent logic ---
def prompt_ai(messages):
    """Handles tool call and returns the tool output directly (no synthesis)."""
    max_history = 15
    system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]

    # Logique de troncature corrigée
    if len(messages) <= max_history:
        recent_msgs = messages
    else:
        # Garder tous les messages système + les derniers messages non-système
        non_system_msgs = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        max_non_system = max(1, max_history - len(system_msgs))  # Au moins 1 message non-système
        recent_msgs = system_msgs + non_system_msgs[-max_non_system:]

    try:
        response = agent.invoke(recent_msgs)

        # Check if the response has valid tool calls
        if not hasattr(response, "tool_calls") or not isinstance(response.tool_calls, list) or not response.tool_calls:
            return response  # No tool call detected, return the agent's response directly

        # Warn the user if multiple tool calls are detected and handle only the first one
        if len(response.tool_calls) > 1:
            return AIMessage(content="⚠️ Multiple tool calls detected. Only the first one will be processed.")

        tool_call = response.tool_calls[0]

        # Safely access 'name' and 'args'
        tool_name = tool_call.get("name", None)

        args = tool_call.get("args", {})

        if not tool_name:
            return AIMessage(content="❌ Invalid tool call: missing 'name' key.")


        question = args.get("question", "").strip()
        if not question:
            return AIMessage(content="❌ Invalid tool call: missing or empty 'question' argument.")

        # Safely select the appropriate tool
        selected_tool = {
            "use_gemini": use_gemini,
            "use_deepseek": use_deepseek,
        }.get(tool_name)

        if not selected_tool:
            return AIMessage(content=f"❌ Unknown tool '{tool_name}'.")

        # Invoke the tool and check its output
        tool_output = selected_tool.invoke(question)

        # If the tool was DeepSeek, clean the output via Gemini
        if tool_name == "use_deepseek":
            tool_output = clean_latex_with_gemini(tool_output)

        # Improved validation with better error messages
        if tool_output is None:
            return AIMessage(content="❌ Tool returned None.")

        if not isinstance(tool_output, str):
            return AIMessage(content=f"❌ Tool returned unexpected type: {type(tool_output)}. Content: {repr(tool_output)[:100]}")

        if not tool_output.strip():
            return AIMessage(content="❌ Tool returned empty string after stripping whitespace.")

        return AIMessage(content=tool_output)

    except Exception as e:
        return AIMessage(content=f"❌ An error occurred during processing: {e}")
