from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv.ipython import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI ,OpenAIEmbeddings
from langchain.messages import HumanMessage
import os
from IPython.display import Markdown
import tokenizers
load_dotenv(override=True)
loader = PyPDFLoader("CV_Radi_IT.pdf")
tokennizer= tiktoken.encoding_for_model("gpt-4o-mini")
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
encoding_name=tokennizer.name, chunk_size=300, chunk_overlap=20
)
chunks= loader.load_and_split(splitter)

print(tokennizer.name)
print(len(chunks))
print(chunks[0].metadata)
print(chunks[0].page_content)
embedding_model= OpenAIEmbeddings()
vecror_store= Chroma.from_documents(
documents=chunks, embedding=embedding_model, 
collection_name="cv_data_collection"
)
retriever = vecror_store.as_retriever(kwargs={"k": 10})
@tool
def retriever_tool(query: str) -> str:
    """
   Permet de chercher des informations sur des candidats :
   -Nom, Prénom, Diplômes
   -Expériences
    -Compétences 
    """
    relevant_chunks= retriever.invoke(query)
    context_list= [d.page_content for d in relevant_chunks]
    context= ". ".join(context_list)
    return context
@tool
def get_company_infos(company_name: str):
    """Consulter des infomrationssur l'entreprise donnée"""
    return{"company_name": company_name, "domain": "IT", "turnover": 120_870_000}
llm= ChatOpenAI(model="gpt-4o-mini", temperature=0)
my_agent= create_agent(
   model=llm,
   tools=[retriever_tool, get_company_infos],
   system_prompt="Répond à la question de l'utilisateur en utilisant les tools fournis",
)
