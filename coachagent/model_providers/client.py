import os

from dotenv import load_dotenv
from openai import OpenAI


def get_openai_client():
    load_dotenv()
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
