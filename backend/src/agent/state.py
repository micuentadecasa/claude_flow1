from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict, Dict, Any, List, Optional

from langgraph.graph import add_messages
from typing_extensions import Annotated

import operator


class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    document_path: str
    questions_status: Dict[str, str]  # question_id -> "answered"|"pending"|"needs_improvement"
    current_question: Optional[str]
    user_context: Dict[str, Any]
    language: str
    conversation_history: List[Dict[str, Any]]


class ReflectionState(TypedDict):
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: Annotated[list, operator.add]
    research_loop_count: int
    number_of_ran_queries: int


class Query(TypedDict):
    query: str
    rationale: str


class QueryGenerationState(TypedDict):
    query_list: list[Query]


class WebSearchState(TypedDict):
    search_query: str
    id: str


@dataclass(kw_only=True)
class SearchStateOutput:
    running_summary: str = field(default=None)  # Final report
