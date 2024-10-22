from llama_index.core.tools import (
    QueryEngineTool,
    ToolMetadata
)
from llama_index.core.workflow import (
    step,
    Context,
    Workflow,
    Event,
    StartEvent,
    StopEvent
)
from llama_index.core.agent import ReActAgent

class QueryEvent(Event):
    question: str

class AnswerEvent(Event):
    question: str
    answer: str

class SubQuestionQueryEngine(Workflow):

    @step(pass_context=True)
    def query(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        