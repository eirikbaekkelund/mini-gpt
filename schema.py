from pydantic import BaseModel
from datetime import datetime
from typing import List

from typing_extensions import Self
class Document(BaseModel):
    doc_id: str
    title: str | None = None
    date_published: datetime | None = None
    date_modified: datetime | None = None
    keywords: List[str] | None = None
    teaser_text: str | None = None
    short_teaser_text: str | None = None
    doc_type: str | None = None
    doc_series: str | None = None
    doc_language: str | None = None
    authors: List[str] | None = None
    license_required: int | None = None
    product: str | None = None
    solution_category: str | None = None
    solution_group: str | None = None
    expiry_date: datetime | None = None
    pa: List[int] | None = None
    csf: List[int] | None = None
    sync_date: datetime | None = None

    @classmethod
    def get_field_names(cls, alias=False):
        return list(cls.model_json_schema(alias).get("properties").keys())


class Chunk(BaseModel):
    uuid: str
    chunk_id: str
    text: str
    document: Document | None
    page: int
    order: int
    nword: int | None = None
    preceding_chunk: Self | None = None
    succeeding_chunk: Self | None = None
    similarity_score: float | None = None
    vector: dict[str, List[float]] | None = None

    @classmethod
    def get_field_names(cls, alias=False):
        return list(cls.model_json_schema(alias).get("properties").keys())
