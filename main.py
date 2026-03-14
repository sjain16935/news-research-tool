import os
import streamlit as st
import time

# ── Env ──────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()
os.environ["USER_AGENT"] = "RockyBot/1.0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"     

# ── LangChain imports ─────────────────────────────────────────────────────────
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.title(" News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

# ─────────────────────────────────────────────────────────────────────────────
# LLM  (phi3:mini via Ollama)
# ─────────────────────────────────────────────────────────────────────────────
llm = OllamaLLM(
    model="phi3:mini",
    temperature=0.9,
    base_url="http://localhost:11434",
    num_predict=200,  # ✅ limits response length = faster
)
# ─────────────────────────────────────────────────────────────────────────────
# Process URLs
# ─────────────────────────────────────────────────────────────────────────────
if process_url_clicked:
    urls_filtered = [u.strip() for u in urls if u.strip()]

    if not urls_filtered:
        st.error("⚠️ Please enter at least one URL.")
        st.stop()

    # 1. Load
    main_placeholder.text("Data Loading... ✅")
    try:
        loader = WebBaseLoader(urls_filtered)
        data = loader.load()
    except Exception as e:
        st.error(f"❌ Failed to load URLs: {e}")
        st.stop()

    if not data:
        st.error("❌ No data loaded. Please check your URLs.")
        st.stop()

    # 2. Split
    main_placeholder.text("Splitting Text... ✅")
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=500,
        chunk_overlap=50,
    )
    docs = splitter.split_documents(data)

    if not docs:
        st.error("❌ No text chunks created. Try different URLs.")
        st.stop()

    # 3. Embed + FAISS
    main_placeholder.text("Building Embeddings... ✅")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434",
    )

    try:
        # Use from_texts to avoid doc.id version mismatch bug
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    except Exception as e:
        st.error(f"❌ Embedding failed: {e}")
        st.stop()

    # 4. Save using FAISS built-in (no pickle)
    vectorstore.save_local("faiss_store")

    time.sleep(1)
    main_placeholder.text("✅ Done! Ask your question below.")

# ─────────────────────────────────────────────────────────────────────────────
# Q&A
# ─────────────────────────────────────────────────────────────────────────────
query = main_placeholder.text_input("Question: ")
if query:
    if not os.path.exists("faiss_store"):
        st.warning("⚠️ Please process URLs first.")
    else:
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434",
        )
        vectorstore = FAISS.load_local(
            "faiss_store",
            embeddings,
            allow_dangerous_deserialization=True,
        )

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
        )

        with st.spinner("Thinking..."):
            try:
                result = chain({"question": query}, return_only_outputs=True)
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.stop()

        st.header("Answer")
        st.write(result.get("answer", "No answer returned."))

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                if source.strip():
                    st.write(source)