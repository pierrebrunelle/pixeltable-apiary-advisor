"""Apiary Advisor: a small RAG + tool-calling agent built with Pixeltable.

Hive-management guides are chunked into sentences, embedded with a custom batched UDF
(feature hashing, no model download) and indexed with an EmbeddingIndex. A @pxt.query
retrieves the best chunks; the Advisor table is an "agent table" whose computed columns
retrieve, format context and draft an answer, and run.py turns that table into a reusable
function with pxt.udf(table, return_value=...). Inspections use an expr_udf for mite load,
and Questions call an LLM with tools (only when OPENAI_API_KEY is set).

Layout: UDFs live in udfs.py; this file holds table models and @pxt.query functions.
"""

from __future__ import annotations

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.functions import openai

from udfs import (
    advisor_messages,
    extractive_answer,
    final_messages,
    format_context,
    hash_embed,
    mite_load_pct,
    varroa_action,
    with_region,
)

TableModel = pxt.model_base()


class Guides(TableModel, name='guides', has_default_idxs=False):
    """One hive-management guide; source_uri may point at object storage or the web."""

    guide_id = pxt.Column(type=pxt.String, primary_key=True)
    title: pxt.String
    topic: pxt.String
    body: pxt.String
    source_uri: pxt.String | None


class GuideChunks(
    TableModel,
    name='guide_chunks',
    base=Guides,
    iterator=pxtf.string.string_splitter(Guides.body, separators='sentence'),
):
    """One row per sentence of a guide, searchable through a custom embedding index."""

    n_chars = pxtf.string.len(text)  # type: ignore[name-defined]  # noqa: F821

    __indexes__ = [
        pxt.EmbeddingIndex(text, string_embed=hash_embed, metric='cosine', name='chunk_embed'),  # type: ignore[name-defined]  # noqa: F821
    ]


@pxt.query
def search_guides(question: str, k: int = 3):
    """Top-k guide sentences for a question (custom-embedding similarity)."""
    sim = GuideChunks.text.similarity(string=question, idx='chunk_embed')
    return (
        GuideChunks.order_by(sim, asc=False)
        .limit(k)
        .select(GuideChunks.guide_id, GuideChunks.title, GuideChunks.text, score=sim)
    )


@pxt.query
def guides_by_topic(topic: str):
    """Guides for one topic, with a shareable link produced by functions.net.presigned_url."""
    return (
        Guides.where(Guides.topic == topic)
        .select(Guides.guide_id, Guides.title, link=pxtf.net.presigned_url(Guides.source_uri, 3600))
        .order_by(Guides.guide_id)
    )


class Advisor(TableModel, name='advisor', has_default_idxs=False):
    """Agent table: question -> retrieval -> context -> offline extractive answer.

    run.py wraps it as a function with pxt.udf(<advisor table>, return_value=...).
    """

    question: pxt.String
    regional_question = with_region(question)
    hits = search_guides(question, 3)
    context = format_context(hits)
    answer = extractive_answer(question, hits)


class Inspections(TableModel, name='inspections', has_default_idxs=False):
    """A hive inspection with a mite sample and a free-text note."""

    inspection_id = pxt.Column(value=pxtf.uuid.uuid7(), primary_key=True)
    hive_id: pxt.String
    mites: pxt.Int
    bees_sampled: pxt.Int
    note: pxt.String

    mite_pct = mite_load_pct(mites, bees_sampled)
    action = varroa_action(mite_pct)

    __indexes__ = [pxt.BtreeIndex(hive_id)]


TOOLS = pxt.tools(varroa_action, search_guides)


class Questions(TableModel, name='questions', has_default_idxs=False):
    """Beekeeper questions answered by an LLM that may call varroa_action / search_guides."""

    question: pxt.String
    hits = search_guides(question, 3)
    context = format_context(hits)
    response = openai.chat_completions(
        advisor_messages(question, context),
        model='gpt-4o-mini',
        model_kwargs={'max_tokens': 160, 'temperature': 0},
        tools=TOOLS,
    )
    tool_output = openai.invoke_tools(TOOLS, response)
    # Keep this model_kwargs literal different from the tool-calling one above: chat_completions
    # adds `tools` to the dict it is given, and equal literals can end up shared between columns.
    final = openai.chat_completions(
        final_messages(question, context, tool_output),
        model='gpt-4o-mini',
        model_kwargs={'max_tokens': 200, 'temperature': 0},
    )
    answer = final.choices[0].message.content


@pxt.query
def hive_history(hive_id: str):
    """Inspections for one hive, newest first (uses the hive_id btree index)."""
    return (
        Inspections.where(Inspections.hive_id == hive_id)
        .select(Inspections.hive_id, Inspections.mites, Inspections.mite_pct, Inspections.action, Inspections.note)
        .order_by(Inspections.inspection_id, asc=False)
    )
