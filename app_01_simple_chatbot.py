"""
This module contains a simple chatbot implementation.
"""

import os
import warnings
from langchain.memory import FileChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain import LLMChain
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain._api import LangChainDeprecationWarning


warnings.simplefilter("ignore", category=LangChainDeprecationWarning)

_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]


chatbot = ChatOpenAI(model="gpt-3.5-turbo")


messagesToTheChatbot = [
    HumanMessage(content="My favorite color is blue."),
]

response = chatbot.invoke(messagesToTheChatbot)

print("\n----------\n")

print("My favorite color is blue.")

print("\n----------\n")
print(response.content)

print("\n----------\n")

response = chatbot.invoke([
    HumanMessage(content="What is my favorite color?")
])

print("\n----------\n")

print("What is my favorite color?")

print("\n----------\n")
print(response.content)

print("\n----------\n")


memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("messages.json"),
    memory_key="messages",
    return_messages=True
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chatbot,
    prompt=prompt,
    memory=memory
)

response = chain.invoke("hello!")

print("\n----------\n")

print("hello!")

print("\n----------\n")
print(response)

print("\n----------\n")

response = chain.invoke("my name is Marcela")

print("\n----------\n")

print("my name is Marcela")

print("\n----------\n")
print(response)

print("\n----------\n")

response = chain.invoke("what is my name?")

print("\n----------\n")

print("what is my name?")

print("\n----------\n")
print(response)

print("\n----------\n")
