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
    prompt = """You are generating test questions for a mathematics AI assistant called EuclidIA.
Generate exactly 10 questions following this structure:

EXPLANATION QUESTIONS (2 questions - should trigger definition/concept explanations):
1. [Question asking for definition or explanation of a math concept]
2. [Question asking for explanation of a mathematical property or theorem - WITH deliberate spelling mistake or typo]

PROOF QUESTIONS (2 questions - should trigger formal mathematical proofs):
3. [Question asking to prove a mathematical theorem]
4. [Question asking to demonstrate a mathematical property - WITH deliberate spelling mistake or typo]

PROBLEM SOLVING (2 questions - should trigger step-by-step calculations):
5. [Concrete math problem requiring calculations, like geometry, algebra, calculus]
6. [Another concrete problem with numerical solution]

ROBUSTNESS TESTS (2 questions - should be rejected as non-mathematical):
7. [Non-mathematical question like weather, politics, etc.]
8. [Impossible mathematical claim like "prove 1=2"]

AMBIGUOUS QUESTIONS (2 questions - should trigger error handling):
9. [Incomplete or unclear mathematical question]
10. [Question with missing information]

IMPORTANT: Return ONLY the numbered list of questions. Do NOT add any comments, notes, explanations, or text in parentheses. Each line should contain ONLY the question number and the question text."""
    try:
        # Update: correct syntax for v1.7.0 with higher temperature for variety
        response = client.chat.complete(
            model="mistral-medium",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8  # Higher temperature for more diverse questions
        )
        lines = response.choices[0].message.content.strip().split("\n")
        questions = [line.lstrip("0123456789. ").strip() for line in lines if line.strip()]

        # Ensure only 10 questions are returned
        return questions[:10]
    except Exception as e:
        print(f"❌ Error generating test questions: {e}")
        # Return fallback questions if API fails
        return [
            "What is the definition of a derivative?",  # Explanation
            "Explain the Pythagorean teorem",  # Explanation with typo
            "Prove that the square root of 2 is irrational",  # Proof
            "Demonstrate the quadrtic formula",  # Proof with typo
            "Calculate the area of a triangle with base 5 and height 8",  # Problem solving
            "Solve the equation 3x² - 7x + 2 = 0",  # Problem solving
            "What is the weather like today?",  # Non-math question
            "Prove that 1 equals 2",  # Impossible claim
            "Calculate the derivative of",  # Incomplete question
            "Find the solution to the equation"  # Missing equation
        ]

# --- Evaluate assistant's answer using Mistral Medium ---
def evaluate_response(question, answer):
    prompt = f"""
You are evaluating the response of a specialized math assistant named EuclidIA.

Rules:
- It must only answer math-related questions.
- It must reject non-math questions politely.
- Math answers must be accurate, clear, and helpful.

Question:
{question}

Answer:
{answer}

Provide your evaluation ONLY in this exact format, enclosed in triple backticks (no extra text):

```
Score: X/10
Comment: <your evaluation>
```

No introduction, no explanations, no greetings, just the two lines inside the code block.
"""
    try:
        # Update: correct syntax for v1.7.0
        response = client.chat.complete(
            model="mistral-medium",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        score_line = next((line for line in content.splitlines() if "Score:" in line), "Score: 0/10")
        comment_line = next((line for line in content.splitlines() if "Comment:" in line), "Comment: No comment.")
        score = float(score_line.split(":")[1].split("/")[0].strip())
        comment = comment_line.split(":", 1)[1].strip()
        return score, comment
    except Exception as e:
        print(f"❌ Error evaluating response: {e}")
        return 0.0, f"Evaluation failed: {e}"

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
    total_score = 0.0
    threshold_score = 7.0  # ✅ Minimum required average score

    for idx, question in enumerate(questions, 1):
        print(f"\n🔹 Q{idx}: {question}")
        try:
            messages = [system_msg, HumanMessage(content=question)]

            # Direct call since prompt_ai now handles everything and returns AIMessage with final content
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
