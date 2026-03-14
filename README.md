# AI News Research Tool 📈

An AI-powered tool that allows users to analyze and ask questions about multiple news articles using LLMs and Retrieval-Augmented Generation (RAG).

## 🚀 Features
- Input multiple news article URLs
- Automatic article scraping
- Semantic search using embeddings
- Ask questions across articles
- Source-backed answers

## 🧠 Tech Stack
- Python
- Streamlit
- LangChain
- Ollama (phi3:mini)
- FAISS Vector Database
- WebBaseLoader

## ⚙️ How It Works
1. User enters news article URLs
2. Articles are scraped and loaded
3. Text is split into chunks
4. Embeddings are generated
5. Data is stored in FAISS vector database
6. LLM retrieves relevant information to answer queries

## ▶️ Run the Project

Clone the repository

```bash
git clone https://github.com/yourusername/news-research-tool.git

Install dependencies

pip install -r requirements.txt

Run the application

streamlit run app.py
📌 Future Improvements

Real-time news monitoring

Multi-document summarization

Better UI


---

### Step 3: Add it to Git

Run these commands:

```bash
git add README.md
git commit -m "Added README file"
git push
