import dotenv
dotenv.load_dotenv()
from llama_agents import (
    AgentService,
    ControlPlaneServer,
    SimpleMessageQueue,
    AgentOrchestrator,
)

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import CouldNotRetrieveTranscript
import urllib.request
import re, os
import numpy as np
import pandas as pd

from llama_index.core import Settings
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
import logging

llm = AzureOpenAI(
    engine = "gpt-35-turbo",
    model = "gpt-35-turbo",
    api_key = os.getenv('AZURE_OPENAI_API_KEY'),  
    api_version = "2024-02-01",
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
)

embed_model = AzureOpenAIEmbedding(
    model = "text-embedding-3-large",
    deployment_name = "text-embedding-3-large",
    api_key = os.getenv('AZURE_OPENAI_API_KEY'),
    api_version = "2024-02-01",
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
)

Settings.llm = llm
Settings.embed_model = embed_model

logging.getLogger("llama_agents").setLevel(logging.INFO)

message_queue = SimpleMessageQueue()
control_plane = ControlPlaneServer(
    message_queue=message_queue,
    orchestrator=AgentOrchestrator(llm=llm),
)

def get_youtube_videos(search_keyword: str) -> list:
    html = urllib.request.urlopen("https://www.youtube.com/results?search_query=" + search_keyword.replace(' ','+'))
    video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())
    unique_ids = []
    for id in video_ids:
        if len(unique_ids) == 10:
            break

        if id not in unique_ids:
            try:
                YouTubeTranscriptApi.get_transcript(id)
                unique_ids.append(id)
            except CouldNotRetrieveTranscript:
                pass
            
    return ["https://www.youtube.com/watch?v="+i for i in unique_ids]

get_youtube_videos_tool = FunctionTool.from_defaults(
    fn=get_youtube_videos,
    description="Returns list of 10 youtube video links for given search keyword."
)

worker = FunctionCallingAgentWorker.from_tools(
    tools=[get_youtube_videos_tool], 
    llm=llm
)
agent = worker.as_agent()

agent_service = AgentService(
    agent=agent,
    message_queue=message_queue,
    description="Purpose: The primary role of this agent is to provide youtube video links for given query. Provide full link in answer.",
    service_name="youtube",
)

from llama_agents import LocalLauncher

launcher = LocalLauncher(
    [agent_service],
    control_plane,
    message_queue,
)

# Run a single query through the system
result = launcher.launch_single("Can you give me 10 videos for Samsung Galaxy S24 Ultra Review?")
print(result)