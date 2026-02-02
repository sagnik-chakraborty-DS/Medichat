from euriai.langchain import create_chat_model
from app.config import API_KEY

"""
temp = closer to 1 more halcunation
and closer to 0 means less halucination
"""

def get_chat_model(model_name : str = "gpt-4.1-nano" , tempareture:float = 0.7 ):
    return create_chat_model(api_key=API_KEY , model_name = model_name,temperature=tempareture)


def ask_chat_model(chat_model , prompt:str):
    response = chat_model.invoke(prompt)
    return response.content