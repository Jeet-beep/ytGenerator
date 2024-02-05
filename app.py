from dotenv import load_dotenv
load_dotenv()
import os
from langchain_community.llms import openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain_openai import ChatOpenAI
import streamlit as st

st.title("Youtube Title Generator App")
prompt = st.text_input("Write your keyword here")

os.environ['OPENAI_API_KEY'] = input("Enter your api key:")
llm = ChatOpenAI(temperature=1.0, model_name="gpt-3.5-turbo")

title_template = PromptTemplate(
    input_variables=["keyword"],
    template="write a creative youtube video title about {keyword}"
)

desc_template = PromptTemplate(
    input_variables=["title"],
    template="based on the {title} of the youtube video, generate a youtube description in almost 150 words and also generate hashtags"
)

title_chain = LLMChain(llm=llm, prompt=title_template, output_key="title")
desc_chain = LLMChain(llm=llm, prompt=desc_template, output_key="description")

sequential_chain = SequentialChain(chains=[title_chain, desc_chain], input_variables=["keyword"],
                                    output_variables=["title", "description"], verbose=True)

if prompt:
    response = sequential_chain({"keyword": prompt})
    st.write(response["title"])
    st.write(response["description"])
