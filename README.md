# 📐 EuclidIA  - *Think. Explain. Prove.*

**EuclidIA** is an AI-powered chatbot specialized in mathematics.  
It answers your questions by explaining clearly or reasoning precisely — just like a real mathematician.

---

### 💡 *Explain with clarity*  
### 🧠 *Reason with precision*  
### ➕ *Focused 100% on mathematics*

---

## ✨ Features

- 📘 **Explains math concepts** — definitions, formulas, and properties  
- 🧠 **Performs reasoning** — for structured mathematical demonstrations

---

## 🚀 Quick Start

```bash
git clone https://github.com/AdelMessaoudi-13/EuclidIA
cd EuclidIA

Add your API keys in .env
Required: GOOGLE_API_KEY, DEEPSEEK_API_KEY

pip install -r requirements.txt
streamlit run app.py
```

Then go to `http://localhost:8501`.

---

## 💬 Examples

> **"What is the Laplace transform?"**  
> → 📘 **Explain clearly**

> **"Prove that √2 is irrational."**  
> → 🧠 **Reason rigorously**

---

## 🛠 Technologies Used

| Type         | Component             | Role                                               |
|--------------|-----------------------|----------------------------------------------------|
| 🧠 LLM        | **Google Gemini 2.5** | Clear and factual math explanations                |
| 🧠 LLM        | **DeepSeek Reasoner** | Structured reasoning and formal demonstrations     |
| 🧩 Framework  | **LangChain**         | Tool orchestration (explain vs. reason)            |
| 🌐 Frontend   | **Streamlit**         | Interactive web interface                          |
| 🔐 Utility    | **python-dotenv**     | Loads API keys from `.env`                         |

---

## 📁 Project Structure

```
euclidia/
├── app.py            ← Streamlit app
├── config.py         ← API keys & model setup
├── tools.py          ← LangChain tools
├── agent_logic.py    ← Agent orchestration logic
├── requirements.txt
├── .env              ← Your API keys
└── README.md
```

---

## 🤝 Contribute

Want to improve or extend EuclidIA?  
Open an issue or a pull request — your contributions are welcome!

---

## 📬 Contact

- ✉️ amessaoudi.am@gmail.com  
- 🌐 [GitHub – AdelMessaoudi-13](https://github.com/AdelMessaoudi-13)  
- 🔗 [LinkedIn](https://www.linkedin.com/in/adel-messaoudi-831358132)  
- 🤗 [Hugging Face](https://huggingface.co/AdelMessaoudi-13)
