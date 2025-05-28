import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek.chat_models import ChatDeepSeek

# --- Load environment variables ---
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# --- API Key validation ---
def check_api_keys():
    if not GOOGLE_API_KEY:
        raise ValueError("❌ GOOGLE_API_KEY is missing.")
    if not DEEPSEEK_API_KEY:
        raise ValueError("❌ DEEPSEEK_API_KEY is missing.")

    # Basic format validation
    if len(GOOGLE_API_KEY.strip()) < 10:
        raise ValueError("❌ GOOGLE_API_KEY appears to be invalid (too short).")
    if len(DEEPSEEK_API_KEY.strip()) < 10:
        raise ValueError("❌ DEEPSEEK_API_KEY appears to be invalid (too short).")

    # Check for common placeholder values
    invalid_values = ["your_api_key", "placeholder", "xxx", "test", ""]
    if GOOGLE_API_KEY.lower().strip() in invalid_values:
        raise ValueError("❌ GOOGLE_API_KEY appears to be a placeholder value.")
    if DEEPSEEK_API_KEY.lower().strip() in invalid_values:
        raise ValueError("❌ DEEPSEEK_API_KEY appears to be a placeholder value.")

class LLMsConfig:
    def __init__(self):
        self.llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", google_api_key=GOOGLE_API_KEY, temperature=0.7)
        self.llm_deepseek = ChatDeepSeek(model="deepseek-reasoner", temperature=0.7)

    def get_llms(self):
        return self.llm_gemini, self.llm_deepseek

# --- Models ---
llms_config = LLMsConfig()
