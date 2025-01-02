import os
from apikey import apikey

os.environ['OPENAI_API_KEY'] = apikey

#streamlit - for building the app
#langchain - for llm workflow
#openai - for openai gpt (generative pre-trained transformer)
#wikipedia - to connect gpt to wikipedia
#chromadb - vector storage
#tiktoken - backend tokenizer for openai

import streamlit as st
from langchain.llms import OpenAI

#App framework
st.title('Youtube video generator')
prompt = st.text_input('Plug in your prompt')

#llm
#llm = OpenAI(temperature = 0.9)

#showing stuff on the screen if theres a prompt
#if prompt:
    #response = llm(prompt)
    #st.write(response)

#adding a prompt by ourselves
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

#this is one chain-
title_template = PromptTemplate(
    input_varibales = ['topic'],
    template='write me a youtube video title about {topic}'
)

#llm (prompt added and theres only one chain in this)
#llm = OpenAI(temperature = 0.9)
#title_chain = LLMChain(llm = llm, prompt = title_template, verbose = True)

#showing stuff on the screen
#if prompt:
    #response = title_chain.run(topic = prompt)
    #st.write(response)

from langchain.chains import SimpleSequentialChain
#to chain mutiple chains together in order to generate multiple outputs

#another chain-
script_template = PromptTemplate(
    input_varibales = ['title'],
    template='write me a youtube video script based on this title TITLE: {title}'
)

#llm (two chains this time)
#llm = OpenAI(temperature = 0.9)
#title_chain = LLMChain(llm = llm, prompt = title_template, verbose = True)
#script_chain = LLMChain(llm = llm, prompt = script_template, verbose = True)
#sequential_chain = SimpleSequentialChain(chains = [title_chain, script_chain], verbose = True)

#showing stuff on the screen
#if prompt:
    #response = sequential_chain.run(prompt)
    #st.write(response)

#to add memory to our application
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(input_key = 'topic', memory_key = 'chat_history')

#simple sequential chain outputs only the last chain's output
#we can use sequential chain instead and specify the output keys
#heres how-


from langchain.chains import SequqntialChain

llm = OpenAI(temperature = 0.9)
title_chain = LLMChain(llm = llm, prompt = title_template, verbose = True, output_key = 'title', memory = memory)
script_chain = LLMChain(llm = llm, prompt = script_template, verbose = True, output_key = 'script', memory = memory)
sequential_chain = SequentialChain(chains = [title_chain, script_chain], verbose = True, output_variables = ['title', 'script'])

#showing stuff on the screen
if prompt:
    response = sequential_chain({'topic':prompt}) #here we can not use the run function, we need to pass a dictionary instead
    st.write(response['title'])
    st.write(response['script'])

    with st.expander('Message History'):
        st.info(memory.buffer)