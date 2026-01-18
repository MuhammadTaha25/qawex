from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_google_genai import GoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore as lang_pinecone
import os
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi.middleware.cors import CORSMiddleware

embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"  # Use the desired HuggingFace model
        )

PINECONE_API_KEY="pcsk_3CsSGf_551ffPDGaVfyL27RwGmdgp5EkBS5AKdMa2QYProa5qdSiao7NiWtzEZh8bpH6Gv"

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

model_name = "gemini-2.5-flash"
LLM = GoogleGenerativeAI(
                model=model_name,
                google_api_key="AIzaSyAaMLduuHncLHZcfdGssIMuQIedC0thkR4"
            )
history = []
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


prompt_str = """
You are QAWEX, the official AI customer support assistant of the QAWEX website.

Your job is to help users clearly, politely, and professionally.
Always respond like a customer service representative.
Keep answers short, friendly, and easy to understand.

Use ONLY the information provided in the context to answer.
If the answer is not available in the context, politely say that QAWEX support will assist further.

Context:
{context}

Conversation History:
{chat_history}

User Question:
{question}

Answer as QAWEX customer support:

Note that:
1. provide response in 1-2 lines
"""

prompt = ChatPromptTemplate.from_template(prompt_str)
question_fetcher=itemgetter("question")
history_fetcher=itemgetter("chat_history")
vector=lang_pinecone.from_existing_index(index_name="qawexpdf", embedding=embeddings)
retriever = vector.as_retriever(search_type="similarity",
                                        search_kwargs={"k": 2})
llm = LLM
setup={
    "question": question_fetcher,
    "chat_history": history_fetcher,
    "context": itemgetter("question") | retriever | format_docs
}
chain = setup|prompt | llm | StrOutputParser()

app = FastAPI()

origins = ["*"]  # ya apni WordPress site URL

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"status": "QAWEX API running"}

@app.post("/chat")
def chat(req: ChatRequest):
    response = chain.invoke({
        "question": req.question,
    })
    return {"answer": response}