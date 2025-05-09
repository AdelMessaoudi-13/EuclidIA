import time
import os
import pandas as pd
from config import llms_config
from agent_logic import prompt_ai
from langchain_core.messages import HumanMessage, SystemMessage
from mistralai import Mistral
from datetime import datetime

# --- Load Mistral API key from environment ---
api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("La variable d'environnement MISTRAL_API_KEY n'est pas définie.")

# Mise à jour: utilisation de Mistral au lieu de MistralClient
client = Mistral(api_key=api_key)

# --- Generate 10 diverse test questions using Mistral Medium ---
def generate_test_questions():
    prompt = """
Generate 10 test questions for evaluating a math assistant:
- 3 clear and correct math questions.
- 2 questions with spelling mistakes in mathematical terms.
- 2 vague or ambiguous math-related questions.
- 2 mathematically incorrect or trick questions.
- 1 off-topic (non-math) question.

Return a numbered list only.
"""
    # Mise à jour: utilisation de la syntaxe correcte pour la v1.7.0
    response = client.chat.complete(
        model="mistral-medium",
        messages=[{"role": "user", "content": prompt}]
    )
    lines = response.choices[0].message.content.strip().split("\n")
    return [line.lstrip("0123456789. ").strip() for line in lines if line.strip()]

# --- Evaluate assistant's answer using Mistral Medium ---
def evaluate_response(question, answer):
    prompt = f"""
You are evaluating the response of a specialized math assistant named EuclidIA.

Its rules:
- It must only answer math-related questions.
- It must reject non-math questions politely.
- Math answers must be accurate, clear, and helpful.

Question:
{question}

Answer:
{answer}

Provide:
Score: X/10
Comment: <your evaluation>
"""
    # Mise à jour: utilisation de la syntaxe correcte pour la v1.7.0
    response = client.chat.complete(
        model="mistral-medium",
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content
    score_line = next((line for line in content.splitlines() if "Score:" in line), "Score: 0/10")
    comment_line = next((line for line in content.splitlines() if "Comment:" in line), "Comment: No comment.")
    score = int(score_line.split(":")[1].split("/")[0].strip())
    comment = comment_line.split(":", 1)[1].strip()
    return score, comment

# --- System message (copied exactly from your code) ---
system_msg = SystemMessage(content="""# 🎯 Role
You are an AI assistant specialized in mathematics. You must answer only questions related to mathematics.

# 🛠️ Available Tools
You have access to two tools to answer questions:
- `use_gemini`: for definitions, clear explanations of mathematical concepts, established properties, formulas, or any factual response.
- `use_deepseek`: for proofs, formal demonstrations, detailed reasoning, or problem solving that requires multiple logical steps.

# 🧭 Guidelines
Carefully analyze each question and choose the most appropriate tool:
- If the question is straightforward, factual, or asks for a simple explanation → use `use_gemini`.
- If the question requires structured reasoning, rigorous justification, or a demonstration → use `use_deepseek`.

Always use only one of these two tools to answer.
""")

# --- Main test runner ---
def run_test_suite():
    print("🚀 Generating test questions using Mistral Medium...")
    questions = generate_test_questions()
    results = []
    total_score = 0

    for idx, question in enumerate(questions, 1):
        print(f"\n🔹 Q{idx}: {question}")
        try:
            messages = [system_msg, HumanMessage(content=question)]
            response = prompt_ai(messages)
            answer = response.content if hasattr(response, "content") else str(response)

            score, comment = evaluate_response(question, answer)
            total_score += score

            print(f"✅ Score: {score}/10 — {comment}")
            results.append({
                "Question": question,
                "Answer": answer,
                "Score": score,
                "Comment": comment
            })

            time.sleep(1)

        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "Question": question,
                "Answer": "[ERROR]",
                "Score": 0,
                "Comment": str(e)
            })

    df = pd.DataFrame(results)
    filename = f"euclidia_test_results_{datetime.now().strftime('%Y-%m-%d')}.csv"
    df.to_csv(filename, index=False)
    avg = total_score / len(questions)
    print(f"\n📄 Results saved to euclidia_test_results.csv")
    print(f"📊 Average score: {avg:.2f}/10")
    print("✅ EuclidIA test completed.\n")

if __name__ == "__main__":
    run_test_suite()
