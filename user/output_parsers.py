from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


model = ChatOllama(
    model = "llama3.2:1b",
    temperature = 0.5,
)

template1 = PromptTemplate(
    template = "Create a detailed report on the topic: {topic}",
    input_variables = ['topic'],
)

template2 = PromptTemplate(
    template = "Write a 5 point summary of text: {text}",
    input_variables = ['text'],
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'blackholes'})

print(result)

# prompt1 = template1.invoke({'topic': 'blackholes'})

# result = model.invoke(prompt1)

# print(f"first detail ------------- {result.content}")

# prompt2 = template2.invoke({'text': result.content})

# result2 = model.invoke(prompt2)

# print(f"final result - {result2.content}")

