import os
from smolagents import OpenAIServerModel, CodeAgent
from .oai_client import aiClient

model = OpenAIServerModel(
    model_id=aiClient.config.model,
    api_base=aiClient.config.base_url,
    api_key=aiClient.config.api_key,
)
agent = CodeAgent(model=model)