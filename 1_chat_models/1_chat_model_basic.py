# Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/
# OpenAI Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/openai/

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model='gpt-4o')

result = model.invoke("What is 81 divided by 2")
print("Full result:")
print(result)
print("Full Content:")
print(result.content)
