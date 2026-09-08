from openai import OpenAI
from dotenv import load_dotenv
import os

# 装载环境变量们，目前是只有OpenAI，但是完全可以拆成一个模块儿，让人可以方便地管理和加载其他服务的环境变量。
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL")

def say_hello():
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input="Briefly introduce yourself including the specific model."
    )
    print(response.output_text)


if __name__ == "__main__":
    say_hello()