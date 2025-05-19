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

    if 'loading_placeholder' in st.session_state:
        st.session_state.loading_placeholder.empty()

    return response.content.strip()

# --- Agent logic ---
def prompt_ai(messages):
    """Handles tool call and returns the tool output directly (no synthesis)."""
    max_history = 15
    system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]
    recent_msgs = system_msgs + messages[-(max_history - len(system_msgs)):] if len(messages) > max_history else messages

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

        if not tool_output or not isinstance(tool_output, str) or not tool_output.strip():
            return AIMessage(content="❌ Tool returned an empty or invalid response.")

        return AIMessage(content=tool_output)

    except Exception as e:
        return AIMessage(content=f"❌ An error occurred during processing: {e}")
