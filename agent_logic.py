import streamlit as st
from tools import use_gemini, use_deepseek
from langchain_core.messages import ToolMessage
from config import llms_config

# --- Load LLM ---
llm_gemini, llm_deepseek = llms_config.get_llms()

def prompt_ai(messages, nested_calls=0):
    if nested_calls > 5:
        raise Exception("Too many nested calls.")

    tools = [use_gemini, use_deepseek]
    agent = llm_gemini.bind_tools(tools)

    # --- Memory trimming ---
    max_history = 15  # Nombre maximum de messages à conserver
    recent_messages = messages[-max_history:] if len(messages) > max_history else messages

    try:
        ai_response = agent.invoke(recent_messages)
    except Exception as e:
        st.error(f"❌ Agent error: {e}")
        #return HumanMessage(content="⚠️ An error occurred while processing your request.")
        from langchain_core.messages import AIMessage
        return AIMessage(content="❌ An error occurred during processing.")

    if hasattr(ai_response, "tool_calls") and ai_response.tool_calls:
        available_tools = {
            "use_gemini": use_gemini,
            "use_deepseek": use_deepseek,
        }

        messages.append(ai_response)

        for tool_call in ai_response.tool_calls:
            tool_name = tool_call["name"]
            args = tool_call["args"]
            selected_tool = available_tools.get(tool_name)

            if selected_tool:
                try:
                    tool_output = selected_tool.invoke(args.get("question", ""))
                except Exception as e:
                    tool_output = f"❌ Tool '{tool_name}' failed: {str(e)}"
                    st.error(tool_output)

                messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))

        return prompt_ai(messages, nested_calls + 1)

    return ai_response
