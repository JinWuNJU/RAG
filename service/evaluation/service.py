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
        self.llm = self._init_llm()
        self.metrics = self._get_available_metrics()

    def _init_llm(self):
        """初始化LLM评估器"""
        return ChatOpenAI(
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            api_key=os.getenv("ZHIPU_API_KEY"),
            model="glm-4-flash"
        )

    def _get_available_metrics(self) -> Dict[str, Dict]:
        """定义可用的RAGAS指标"""
        return {
            "answer_relevancy": {
                "name": "答案相关性",
                "description": "衡量答案与问题的相关程度",
                "implementation": answer_relevancy,
                "llm_wrapper": LangchainLLMWrapper(llm=self.llm)
            },
            "faithfulness": {
                "name": "答案忠实度",
                "description": "衡量答案是否忠实于提供的上下文",
                "implementation": faithfulness,
                "llm_wrapper": LangchainLLMWrapper(llm=self.llm)
            }
        }

    async def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        metric_ids: List[str]
    ) -> Dict:
        """
        执行RAGAS评估
        :return: {
            "scores": [0.8, 0.9, ...],
            "details": [
                {"question": "...", "answer": "...", "score": 0.8, "reason": "..."},
                ...
            ]
        }
        """
        selected_metrics = [
            self.metrics[metric_id]["implementation"](llm=self.metrics[metric_id]["llm_wrapper"])
            for metric_id in metric_ids
            if metric_id in self.metrics
        ]

        if not selected_metrics:
            raise ValueError("未选择有效的评估指标")

        dataset_dict = {
            "question": questions,
            "answer": answers
        }

        result = await evaluate(dataset_dict, metrics=selected_metrics)
        return self._format_results(result, questions, answers)

    def _format_results(self, result, questions, answers) -> Dict:
        """格式化评估结果"""
        return {
            "scores": result.scores.tolist(),
            "details": [
                {
                    "question": q,
                    "answer": a,
                    "score": s,
                    "details": {
                        "reason": "符合评估标准" if s > 0.7 else "需要改进"
                    }
                }
                for q, a, s in zip(questions, answers, result.scores)
            ]
        }