"""Pydantic models exchanged between the stages of the RAG pipeline."""

import uuid

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A span of characters inside one file of the corpus."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without an answer."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its reference sources and answer."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of RAG questions."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """The sources retrieved for one question."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """The sources retrieved for one question, with the generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Search results for a whole dataset."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Search results and answers for a whole dataset."""

    search_results: list[MinimalAnswer]
    k: int


class Chunk(MinimalSource):
    """A piece of a file produced by the chunking stage."""


class IndexData(BaseModel):
    chunks: list[Chunk]