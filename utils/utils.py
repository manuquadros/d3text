from typing import Optional

from datamodel import HtmlArticle, Response
from pydantic import EmailStr
from tokenizers.normalizers import BertNormalizer

from xmlparser import split_metadata_body


def get_batch(annotator_email: EmailStr, batch_size: int) -> list[HtmlArticle]:
    chunks = []
    for item in db.unannotated(annotator_email, batch_size):
        chunks.append(response_to_article(item))

    return chunks


def response_to_article(item: Response) -> HtmlArticle:
    content = BertNormalizer(lowercase=False).normalize_str(item.content)
    metadata, body = split_metadata_body(content)

    return HtmlArticle(
        article_id=item.article.id,
        chunk_id=item.chunk.id,
        metadata=metadata,
        body=body,
    )


def safe_concat(string: Optional[str], suffix: Optional[str]) -> str | None:
    match (string, suffix):
        case (None, None):
            return None
        case (s, None) | (None, s):
            return s
        case (s, t):
            return s + t
