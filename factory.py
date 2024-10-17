from typing import List
from llama_index.core import Document, Settings
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.tools import FunctionTool
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import CouldNotRetrieveTranscript
import urllib.request
import re, os
import numpy as np
import pandas as pd
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from dotenv import load_dotenv
load_dotenv()

local_llm = Ollama(
    model="mistral", 
    request_timeout=60.0
)

local_embed_model = OllamaEmbedding(
    model_name="mxbai-embed-large",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

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


def youtube_search_agent_factory() -> FunctionCallingAgent:
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

    return FunctionCallingAgent.from_tools(
        tools=[get_youtube_videos_tool],
        llm=llm,
        verbose=False,
        system_prompt="Purpose: The primary role of this agent is to provide youtube video links for given query. Provide full link in answer."
    )

def summarize_youtube_agent_factory() -> FunctionCallingAgent:
    def summarize_text(text: str) -> str:
        prompt = f"Summarize the following text in a concise way:\n\n{text}"
        
        response = local_llm.chat([
            ChatMessage(role="user",content=prompt)
        ])
        
        return response.message.content
    
    def youtube_links_to_summary(youtube_links: List[str]) -> bool:
        loader = YoutubeTranscriptReader()
        documents = loader.load_data(ytlinks=youtube_links)
        df = pd.DataFrame([{'doc_id':doc.doc_id,'text':summarize_text(doc.text)} for doc in documents])
        df.to_csv("top_10_summaries.csv",index=False, sep="|")
        return True

    summarize_youtube_video_tool = FunctionTool.from_defaults(
        fn=youtube_links_to_summary,
        description="Returns True if summaries are stored in file."
    )

    return FunctionCallingAgent.from_tools(
        tools=[summarize_youtube_video_tool],
        llm=llm,
        verbose=False,
        system_prompt="Purpose: The primary role of this agent is to save summary of each youtube video link in csv file."
    )

def similarity_agent_factory() -> FunctionCallingAgent:
    def cosine_similarity(vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    def get_top_n_similar_videos(top_n: int) -> List[dict]:
        df = pd.read_csv("top_10_summaries.csv",sep="|")
        documents = []
        for t in df.itertuples():
            documents.append(
                Document(
                    doc_id=t.doc_id,
                    text=t.text,
                    metadata={'video_id':t.doc_id},
                    embedding=local_embed_model.get_text_embedding(t.text)
                )
            )

        sim_matrix = np.zeros(shape=(10,10))
        for i in range(len(documents)):
            for j in range(i+1,len(documents)):
                sim_matrix[i][j] = cosine_similarity(documents[i].embedding,documents[j].embedding)
        
        flat_sim_matrix = sim_matrix.flatten()
        indices = np.argpartition(flat_sim_matrix, -top_n)[-top_n:]
        indices = np.flip(indices[np.argsort(flat_sim_matrix[indices])])

        most_similar_pairs = []
        for idx in indices:
            max_idx = np.unravel_index(idx,sim_matrix.shape)
            most_similar_pairs.append({
                'video1': "https://www.youtube.com/watch?v="+documents[max_idx[0]].doc_id,
                'video2': "https://www.youtube.com/watch?v="+documents[max_idx[1]].doc_id,
                'similarity': sim_matrix[max_idx[0]][max_idx[1]]
            })
        
        return most_similar_pairs

    most_similar_videos_tool = FunctionTool.from_defaults(
        fn=get_top_n_similar_videos,
        description="Returns list of pairs of youtube links with most similarity."
    )

    return FunctionCallingAgent.from_tools(
        tools=[most_similar_videos_tool],
        llm=llm,
        verbose=True,
        system_prompt="Purpose: The primary role of this agent is to give n pairs of most similar youtube videos."
    )