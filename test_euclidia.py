import time
import os
import pandas as pd
from agent_logic import prompt_ai
from langchain_core.messages import HumanMessage, SystemMessage
from mistralai import Mistral
from datetime import datetime

# --- Load Mistral API key from environment ---
api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("Environment variable MISTRAL_API_KEY is not defined.")

# Update: using Mistral instead of MistralClient
client = Mistral(api_key=api_key)

# --- Generate 10 diverse test questions using Mistral Medium ---
def generate_test_questions():
    prompt = """
Generate 10 diverse test questions to evaluate a math assistant capable of both explaining and reasoning:

- 2 mathematical questions requiring a factual answer or definition.
- 2 mathematical questions requiring a proof, demonstration, or detailed logical reasoning.
- 2 vague or ambiguous math-related questions that could be addressed either by a simple explanation or by detailed reasoning.
- 2 tricky or deceptive mathematical problems that require careful step-by-step analysis.
- 1 question containing intentional spelling mistakes in mathematical terms.
- 1 off-topic (non-math) question to test the assistant's ability to politely reject irrelevant queries.

Return a clean, numbered list only.
"""
    # Update: correct syntax for v1.7.0
    response = client.chat.complete(
        model="mistral-medium",
        messages=[{"role": "user", "content": prompt}]
    )
    lines = response.choices[0].message.content.strip().split("\n")
    questions = [line.lstrip("0123456789. ").strip() for line in lines if line.strip()]

    # Ensure only 10 questions are returned
    return questions[:10]
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
    # Update: correct syntax for v1.7.0
    response = client.chat.complete(
        model="mistral-medium",
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content
    score_line = next((line for line in content.splitlines() if "Score:" in line), "Score: 0/10")
    comment_line = next((line for line in content.splitlines() if "Comment:" in line), "Comment: No comment.")
    score = score_line.split(":")[1].split("/")[0].strip()
    comment = comment_line.split(":", 1)[1].strip()
    return score, comment

# --- System message (copied exactly from your code) ---
system_msg = SystemMessage(content="""# Role
You are an AI assistant specialized in mathematics. You must answer only questions related to mathematics.

# Available Tools
You have access to two tools to answer questions:
- `use_gemini`: for definitions, clear explanations of mathematical concepts, established properties, formulas, or any factual response.
- `use_deepseek`: for proofs, formal demonstrations, detailed reasoning, or problem solving that requires multiple logical steps.

# Guidelines
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
    threshold_score = 7  # ✅ Minimum required average score

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

    avg = total_score / len(questions)

    # Add average score to the CSV file
    results.append({
        "Question": "Average",
        "Answer": "",
        "Score": avg,
        "Comment": "Average score over all questions"
    })

    df = pd.DataFrame(results)
    filename = f"euclidia_test_results_{datetime.now().strftime('%Y-%m-%d')}.csv"
    df.to_csv(filename, index=False)
    print(f"\n📄 Results saved to euclidia_test_results.csv")
    print(f"📊 Average score: {avg:.2f}/10")
    print("✅ EuclidIA test completed.\n")

    # Raise error if average score is too low
    if avg < threshold_score:
        raise RuntimeError(f"❌ Test FAILED — average score {avg:.2f}/10 is below the threshold ({threshold_score}/10)")

if __name__ == "__main__":
    run_test_suite()
