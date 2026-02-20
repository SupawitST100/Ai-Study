from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# =========================
# โหลด PDF
# =========================
loader = PyPDFLoader("Nexus_Scripts_Complete_Guide.pdf")
documents = loader.load()

# แบ่งข้อความ
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# สร้าง embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# สร้าง vector store
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# =========================
# Flask
# =========================
app = Flask(__name__)
CORS(app)

groq_api_key = os.getenv("GROQ_API_KEY")
print("KEY:", groq_api_key)

model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
)

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")

    # 🔥 แปลเป็นอังกฤษเพื่อค้นหา
    translated_query = model.invoke(
        f"Translate this to English for document search only: {user_message}"
    ).content

    relevant_docs = retriever.invoke(translated_query)

    if not relevant_docs:
        return jsonify({
            "response": "I couldn't find that information in our documentation."
        })

    context = "\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
You are Nexus Scripts official support AI.

- Answer in the SAME language as the customer's question.
- Use ONLY the documentation below.
- If not found, say you don't have that information.

DOCUMENTATION:
{context}

Customer Question:
{user_message}
"""

    response = model.invoke(prompt).content

    return jsonify({"response": response})


# =========================
# Run Server
# =========================
if __name__ == "__main__":
    app.run(debug=True, port=5000)