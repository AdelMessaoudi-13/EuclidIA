import streamlit as st
from tools import use_gemini, use_deepseek
from config import llms_config
import json

# --- Load LLM ---
llm_gemini, llm_deepseek = llms_config.get_llms()

def prompt_ai(messages):
    tools = [use_gemini, use_deepseek]
    agent = llm_gemini.bind_tools(tools)

    max_history = 15
    recent_messages = messages[-max_history:] if len(messages) > max_history else messages

    # Log the recent messages sent to the agent
    #print("[DEBUG] Sending messages to agent:")
    #for msg in recent_messages:
    #    print(f"  - [{msg.type}] {msg.content}")

    try:
        ai_response = agent.invoke(recent_messages)
        #print("[DEBUG] Agent response received.")
        return ai_response
    except Exception as e:
        #print(f"[ERROR] Agent invocation failed with: {e}")
        from langchain_core.messages import AIMessage
        return AIMessage(content="❌ An error occurred during processing.")
