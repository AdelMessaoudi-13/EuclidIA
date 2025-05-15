from tools import use_gemini, use_deepseek
from config import llms_config
from langchain_core.messages import AIMessage, SystemMessage

# --- Load LLMs & bind once ---
llm_gemini, llm_deepseek = llms_config.get_llms()
tools = [use_gemini, use_deepseek]
agent = llm_gemini.bind_tools(tools)

def prompt_ai(messages):
    """Handles tool call and returns the tool output directly (no synthesis)."""
    max_history = 15
    system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]
    recent_msgs = system_msgs + messages[-(max_history - len(system_msgs)):] if len(messages) > max_history else messages

    #print("\n[AGENT] 📩 Invoking agent with recent messages:")
    #for msg in recent_msgs:
        #print(f"   - [{msg.type}] {msg.content}")

    try:
        response = agent.invoke(recent_msgs)
        #print(f"[AGENT] 🤖 Agent responded with: {response.content}")

        if not (hasattr(response, "tool_calls") and response.tool_calls):
            #print("[AGENT] ✅ No tool call detected, returning response directly.")
            return response

        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        question = tool_call["args"].get("question", "")

        #print(f"[AGENT] 🛠 Tool call detected: {tool_name} with question: {question}")

        selected_tool = {
            "use_gemini": use_gemini,
            "use_deepseek": use_deepseek,
        }.get(tool_name)

        if not selected_tool:
            #print(f"[AGENT] ❌ Unknown tool '{tool_name}'.")
            return AIMessage(content=f"❌ Unknown tool '{tool_name}'.")

        # Invoke tool and return its output directly
        tool_output = selected_tool.invoke(question)
        #print(f"[AGENT] 🔧 Tool '{tool_name}' returned:\n{tool_output}")

        # Direct return as AIMessage (no synthesis)
        return AIMessage(content=tool_output)

    except Exception as e:
        #print(f"[AGENT] ❌ An error occurred during processing: {e}")
        return AIMessage(content=f"❌ An error occurred during processing: {e}")
