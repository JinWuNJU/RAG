import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

load_dotenv()
# 查看 https://docs.ragas.io/en/stable/getstarted/evals/#evaluating-using-a-llm-based-metric
llm = ChatOpenAI(
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key=os.getenv("ZHIPU_API_KEY"),
        model="glm-4-flash"
)
evaluator_llm = LangchainLLMWrapper(llm)