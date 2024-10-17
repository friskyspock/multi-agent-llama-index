import os
from llama_index.core.agent import ReActAgent
from llama_index.llms.azure_openai import AzureOpenAI
from factory import youtube_search_agent_factory, summarize_youtube_agent_factory, similarity_agent_factory

from dotenv import load_dotenv
load_dotenv()

llm = AzureOpenAI(
    engine = "gpt-35-turbo",
    model = "gpt-35-turbo",
    api_key = os.getenv('AZURE_OPENAI_API_KEY'),  
    api_version = "2024-02-01",
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
)

orchestrator = ReActAgent.from_tools(
    tools=[youtube_search_agent_factory, summarize_youtube_agent_factory, similarity_agent_factory],
    llm=llm,
    verbose=True,
    context="You are on orchestration agent. Your job is to decide which agent to run based on the current state of the user and what they've asked to do. Agents are identified by tools."
)

orchestrator.query9