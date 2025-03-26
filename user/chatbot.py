from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


load_dotenv()

model = ChatOpenAI(model_name = "gpt-4o-mini", temperature = 0.9)

chat_history = [SystemMessage("You are a helpful but funny assistent.")]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(result.content))
    print(f'Bot: {result.content}')


print(chat_history)