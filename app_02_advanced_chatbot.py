"""
This module contains a advanced chatbot implementation.
"""

import os
import warnings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
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


chatbotMemory = {}

# input: session_id, output: chatbotMemory[session_id]


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chatbotMemory:
        chatbotMemory[session_id] = ChatMessageHistory()
    return chatbotMemory[session_id]


chatbot_with_message_history = RunnableWithMessageHistory(
    chatbot,
    get_session_history
)

session1 = {"configurable": {"session_id": "001"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="My favorite color is black.")],
    config=session1,
)

print("\n----------\n")

print("My favorite color is black.")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="What's my favorite color?")],
    config=session1,
)

print("\n----------\n")

print("What's my favorite color? (in session1)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

session2 = {"configurable": {"session_id": "002"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="What's my favorite color?")],
    config=session2,
)

print("\n----------\n")

print("What's my favorite color? (in session2)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

session1 = {"configurable": {"session_id": "001"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="What's my favorite color?")],
    config=session1,
)

print("\n----------\n")

print("What's my favorite color? (in session1 again)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

session2 = {"configurable": {"session_id": "002"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="Mi name is Marcela.")],
    config=session2,
)

print("\n----------\n")

print("Mi name is Marcela. (in session2)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="What is my name?")],
    config=session2,
)

print("\n----------\n")

print("What is my name? (in session2)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="What is my favorite color?")],
    config=session1,
)

print("\n----------\n")

print("What is my favorite color? (in session2)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")


def limited_memory_of_messages(messages, number_of_messages_to_keep=2):
    return messages[-number_of_messages_to_keep:]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

limitedMemoryChain = (
    RunnablePassthrough.assign(
        messages=lambda x: limited_memory_of_messages(x["messages"]))
    | prompt
    | chatbot
)

chatbot_with_limited_message_history = RunnableWithMessageHistory(
    limitedMemoryChain,
    get_session_history,
    input_messages_key="messages",
)

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="My favorite vehicles are Vespa scooters.")],
    config=session1,
)

print("\n----------\n")

print("My favorite vehicles are Vespa scooters. (in session1)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="My favorite city is San Francisco.")],
    config=session1,
)

print("\n----------\n")

print("My favorite city is San Francisco. (in session1)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_limited_message_history.invoke(
    {
        "messages": [HumanMessage(content="what is my favorite color?")],
    },
    config=session1,
)

print("\n----------\n")

print("what is my favorite color? (chatbot with memory limited to the last 3 messages)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="what is my favorite color?")],
    config=session1,
)

print("\n----------\n")

print("what is my favorite color? (chatbot with unlimited memory)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")
