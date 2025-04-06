import os
import json
from uuid import UUID
from typing import List, Dict
from sqlalchemy.orm import Session
from loguru import logger
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_community.chat_models import ChatOpenAI

class EvaluationService:
    def __init__(self, db: Session):
        self.db = db
        self.llm = ChatOpenAI(
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            api_key=os.getenv("ZHIPU_API_KEY"),
            model="glm-4-flash"
        )
        self.evaluator_llm = LangchainLLMWrapper(self.llm)

        self.metrics = {
            "answer_relevancy": {
                "name": "答案相关性",
                "description": "衡量答案与问题的相关程度",
                "implementation": answer_relevancy
            },
            "faithfulness": {
                "name": "答案忠实度",
                "description": "衡量答案是否忠实于提供的上下文",
                "implementation": faithfulness
            }
        }

    async def evaluate_dataset(self, questions: List[str], answers: List[str], metric_names: List[str]):
        """执行RAGAS评估"""
        metrics = [self.metrics[name]["implementation"] for name in metric_names]
        return await evaluate(questions=questions, answers=answers, metrics=metrics)