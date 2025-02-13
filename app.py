import bm25s
import Stemmer  # optional: for stemming
import re
import time
import base64
import requests
import trafilatura
from langchain_community.tools import DuckDuckGoSearchResults,DuckDuckGoSearchRun
from bs4 import BeautifulSoup
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer,util
from itertools import cycle
from gtts import gTTS
from io import BytesIO
import requests
#from melo.api import TTS
from googlesearch import search
import asyncio
import chromadb
client = chromadb.PersistentClient(path="/home/bhanu/Downloads/FDA_Data")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
###########################################################################################################
st.set_page_config(layout="wide")
st.title("Virtual AI Pharmacist by BioNeural AI")
st.write("A smarter way to chat with Virtual AI Pharmacist. Redefining human-AI interaction.")

with st.sidebar:
    st.write("Designed & Developed by Bhanu Prakash Doppalapudi")
    st.write("Connect with me on ")
    column1, column2 = st.columns(2)
    column1.markdown(
    """<a href="https://www.linkedin.com/in/bhanu-prakash-doppalapudi-38b49489">
    <img src="data:image/png;base64,{}" width="25">
    </a>""".format(
        base64.b64encode(open("linkedin.png", "rb").read()).decode()
    ),
    unsafe_allow_html=True,
    )
    column2.markdown(
    """<a href="https://github.com/bhanuprakashd/">
    <img src="data:image/png;base64,{}" width="25">
    </a>""".format(
        base64.b64encode(open("github.png", "rb").read()).decode()
    ),
    unsafe_allow_html=True,
    )




# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{'role':'assistant','content':"Hello Human!!:)"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

if len(st.session_state.chat_history)>12:
    st.session_state.chat_history=st.session_state.chat_history[-1:]


print(len(st.session_state.chat_history))



########################

llm = ChatOpenAI(
        model = "qwen2.5:7b",
    base_url = "http://localhost:11434/v1",
    openai_api_key='NA',
    streaming=False)



def history_similarity(query):
    if len(st.session_state.chat_history)>1:
        embeddings = model.encode(st.session_state.chat_history)
        q_embd = model.encode(query)
        scores = util.semantic_search(q_embd,embeddings)[0]
        return scores[0]['score']
    else:
        return 0.0

########################
#speed =1.0
#device ='cpu'
#tts_model = TTS(language='EN', device=device)
#speaker_ids = tts_model.hps.data.spk2id
# British accent
#output_path = 'en-br.wav'

from PIL import Image
import ollama
import tempfile
import os
import ast

from keybert import KeyBERT
kw_model = KeyBERT()
from transformers import pipeline
#pipe = pipeline("text-classification", model="wesleyacheng/news-topic-classification-with-bert")
pipe = pipeline("text-classification", model="dstefa/roberta-base_topic_classification_nyt_news",device='cpu')
model_name = "BAAI/bge-m3"

model = SentenceTransformer(model_name,device='cpu')
shield_llm = ChatOpenAI(
    api_key="ollama",
    model="llama-guard3:1b",
    base_url="http://localhost:11434/v1",
    streaming=False
)

stool = DuckDuckGoSearchResults()
quick_search = DuckDuckGoSearchRun()


def generate_text(instruction, file_path):
    result = ollama.generate(
        model='llava:7b',
        prompt=instruction,
        images=[file_path],
        stream=False
    )['response']
    return result

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1500,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

def remove_brackets(text):
    # Replace the brackets with an empty string
    text = text.replace('[', '').replace(']', '').replace(',','')
    return text
    
def extract_hyperlinks(text):
    
    # This regular expression pattern matches most URLs
    pattern = r'(https?://\S+)'
    # Find all matches in the text
    links = re.findall(pattern, text)
    return [remove_brackets(x) for x in links]
def extract_text_from_url(url):
    downloaded = trafilatura.fetch_url(url)  # Replace with the URL
    extracted_text = trafilatura.extract(downloaded)
    return extracted_text

def extract_data(url):

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    response = requests.get(url,headers =headers)

    soup = BeautifulSoup(response.text, 'html.parser')

    return soup.get_text()

def get_data(found_urls):
    data = []
    for url in found_urls:
        data.append(extract_data(url))
    result = " ".join(data)
    texts = text_splitter.split_text(result)
    return texts


def recommendations(listy,urls,topic):
    embeddings = model.encode(listy)
    topic_embedding = model.encode(topic)
    scores = util.semantic_search(topic_embedding,embeddings,top_k=4)[0]
    recommendations = [urls[item['corpus_id']] for item in scores]
    titles = [listy[item['corpus_id']] for item in scores]
    return recommendations,titles

def image_search(keywords):
    """ Image search """
    images=[]
    titles=[]
    for key in keywords:
        results = DDGS().images(
        keywords=key,
        region="us-en",
        safesearch="off",
        max_results=1)

        for item in results:
            images.append(item['image'])
            titles.append(item['title'])
    #imgs,desc = recommendations(titles,images,topic)
    return images,titles

def videos_search(keywords):
    """ Video Search """
    videos=[]
    titles=[]
    for key in keywords:
        results = DDGS().videos(
        keywords=key,
        region="us-en",
        safesearch="off",
        timelimit="m",
        resolution="high",
        duration="medium",
        max_results=1,
        )
    
        for item in results:
            titles.append(item['description'])
            videos.append(item['content'])
    #ideos,desc = recommendations(descriptions,urls,topic)
    return videos,titles


def return_contexts(query,corpus):

    corpus = [x for x in corpus if x!=None] 
    
    # optional: create a stemmer
    stemmer = Stemmer.Stemmer("english")

    # Tokenize the corpus and only keep the ids (faster and saves memory)
    query_tokens = bm25s.tokenize(query, stemmer=stemmer)
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

    # Create the BM25 model and index the corpus
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
    if len(corpus)>=5: 
        results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=5)
        context = "\n".join(results[0])
    else:
        context="No context available."
    return context

def clean_text(text):
    text =text.replace("*",'')
    return text
hazard_categories = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections"
}

def reformulate_query(query,chat_history):
    history_prompt=f"""'Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.\n\nQuestion:{query}\n\nChat_History:{chat_history}\n\nAnswer:"""
    result = llm.invoke(history_prompt)
    return result.content      


# Accept user input
if prompt := st.chat_input("What is up?"):
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        
        with st.spinner("Thinking.."):
            
            prompt_type = pipe(prompt)[0]['label']
            safety = shield_llm.invoke(prompt)
            safety_list = safety.content.split('\n')
            
            score = history_similarity(prompt)
            if score>0.50:
                prompt = reformulate_query(prompt,"\n".join(st.session_state.chat_history))

            if safety_list[0]=='safe' and prompt_type == "Health and Wellness":
                
                context = requests.post(url='http://127.0.0.1:9000/get_rag_context',json={"query":prompt})
              
                prompt_llm=f"""You are an AI Pharmacist designed by BioNeural AI to engage users in natural, human-like conversations for providing medicine information. If you dont know the answer simply say I dont have an answer. If you find proper medicine related answer add a tag '**FDA Approved Content**'\n\nQuestion:{prompt}\\n\nContext:{context}n\nAnswer:"""
                result = llm.invoke(prompt_llm)
                response = result.content
                if  response.startswith("I don"):
                    results = []
                    urls = search(prompt, num_results = 15)
                    for item in urls:
                        results.append(item)
                    data=[]
                    for item in results:
                        text = extract_data(item)
                        text=" ".join(text.replace("\n\n\n\n\n","").split())
                        data.append(text)
                    
                    result = " ".join(data)
                    corpus = text_splitter.split_text(result)
                    deep_context = return_contexts(prompt,corpus)
                    prompt_llm = "You are an AI Pharmacist designed designed by BioNeural AI  to engage users in natural, human-like conversations for providing medicine information. If you dont know the answer simply say I dont have an answer.  \n\nContext: "+deep_context+"\n\nQuestion: "+prompt+"\n\nAnswer:"
                    stream = llm.invoke(prompt_llm)
                    response = stream.content
                    st.session_state.chat_history.append("user:"+prompt+"assistant:"+response)
                else:
                    #print("INSIDE-2")
                    st.write(response)
                    st.session_state.chat_history.append("user:"+prompt+"assistant:"+response)
            
            elif safety_list[0]=='unsafe' and safety_list[1]!='S6':
                print("INSIDE-2")
                response = "The prompt is in violation of safety standards set by BioNeural AI for safeguarding AI Virtual Pharmacist which falls under hazard category of "+hazard_categories[safety_list[1]]
                st.write(response)
            elif safety_list[0]=='safe' :
                print("INSIDE-3")
                prompt_llm = "You are an AI Pharmacist created  and designed by Bhanu Prakash Doppalapudi ('https://www.linkedin.com/in/bhanu-prakash-doppalapudi-38b49489/'') from BioNeural AI (https://bioneural.netlify.app/) in collaboration with Alaparthi Babu Rao Associates to answer Medicine related information only. Strictly Do not answer  any other  Information. \n\nQuestion:"+prompt+"\n\nAnswer:"
                response = llm.invoke(prompt_llm).content
                st.write(response.replace("Qwen","Virtual AI Pharmacist").replace("Alibaba Cloud","BioNeural AI created by Bhanu Prakash Doppalapudi in collaboration with Alaparthi Babu Rao Associates"))
                st.session_state.chat_history.append("user:"+prompt+"assistant:"+response) 
            
   
            st.balloons()
        
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        


