 
#Wikipedia Query Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper,description="Use this tool for general knowledge questions about people, places, and historical events")
  

 

#%pip install --upgrade --quiet  langchain-google-genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("OFF_Campus.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
Documents = text_splitter.split_documents(docs)

vectordb = FAISS.from_documents(Documents, GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"))

retriver = vectordb.as_retriever()
 

from langchain.tools.retriever_tool import create_retriever_tool

retrieval_tool = create_retriever_tool(
    retriever=retriver, name="Document_Search", description="useful to know Daphal Sanket Anil"
)


#ARXIV Tool
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper   

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper,description = "Use this tool ONLY for highly technical questions about computer science, physics, or machine learning from scientific research papers.")
 

tools = [retrieval_tool, wiki, arxiv]

from dotenv import load_dotenv
load_dotenv() # This correctly loads the key from your .env file into the environment
import os
# The hardcoded line that was here is now gone
from langchain_google_genai import ChatGoogleGenerativeAI  


 


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

#promt template
from langchain import hub

## get promt template from langchain hub
prompt = hub.pull("hwchase17/openai-functions-agent")
 

##Agent
from langchain.agents import create_tool_calling_agent
agent = create_tool_calling_agent(llm, tools, prompt=prompt)
 
 
 # To run agent we need agent executer
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#streamlit framework
import streamlit as st 
st.title("RAG Pipeline with Multiple Data Sources")
st.subheader("by - Sanket Daphal")    
st.text("You can ask me about Sanket, or search seamlessly across Wikipedia and Arxiv — all in one place")
 
input_text = st.text_input("Enter your question here")

flag = False
flag2 = False
if input_text:
    response = agent_executor.invoke({"input": input_text})
    st.write(response['output'])
    flag = True
 
 
    
if flag:
    st.write("Are you satisfy with the response?")
    if st.button("Yes"):
        st.write("Great! Glad you are satisfied with the response.")
    if st.button("No"):
        feedback = st.text_area("Please provide your feedback here")
        if feedback:
            st.write("Thank you for your feedback!")     

    

#agent_executor.invoke({"input": "what is langchain?"})





