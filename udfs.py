"""Pixeltable UDFs for the apiary-advisor RAG agent (recorded as `udfs.<name>`).

Pure Python, no table imports. app.py imports these; tables and @pxt.query live in app.py.
"""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np

import pixeltable as pxt
from pixeltable.func import Batch

EMBED_DIM = 256
_TOKEN_RE = re.compile(r'[a-z0-9]+')
_STOP = frozenset(
    'a an and are as at be by for from has have how i in is it its of on or should the to was what when '
    'which with my our we do does can this that'.split()
)

# A database variable: its value comes from the [[pixeltable.database]] entry in pixeltable.toml
# (vars.apiary_region) or from the PIXELTABLE_VAR_APIARY_REGION environment variable.
APIARY_REGION = pxt.ConfigVar('apiary_region', str)


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or '').lower()) if t not in _STOP]


def _hash_vector(text: str) -> np.ndarray:
    """Feature-hashed bag of words + bigrams, L2-normalized (deterministic, no model download)."""
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    toks = _tokens(text)
    grams = toks + [f'{a}_{b}' for a, b in zip(toks, toks[1:])]
    for g in grams:
        h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), 'little')
        sign = 1.0 if (h >> 63) & 1 else -1.0
        vec[h % EMBED_DIM] += sign * (1.0 if '_' not in g else 0.5)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


@pxt.udf(batch_size=16)
def hash_embed(texts: Batch[str]) -> Batch[pxt.Array[(EMBED_DIM,), pxt.Float]]:
    """Custom embedding UDF for the chunk index (batched, 256-d)."""
    return [_hash_vector(t) for t in texts]


@pxt.expr_udf
def mite_load_pct(mites: pxt.Int, bees_sampled: pxt.Int):
    """Mites per 100 bees, built as an expression template (inlined into the query plan)."""
    return mites * 100.0 / bees_sampled


@pxt.udf
def varroa_action(mites_per_100: float) -> str:
    """Map a mite wash/sugar-roll count (mites per 100 bees) to a management action."""
    if mites_per_100 is None or math.isnan(mites_per_100):
        return 'no sample: do an alcohol wash of ~300 bees'
    if mites_per_100 < 1.0:
        return 'low: re-sample in 4 weeks'
    if mites_per_100 < 3.0:
        return 'rising: plan treatment, re-sample in 2 weeks'
    return 'above threshold: treat now and re-sample after treatment'


@pxt.udf
def with_region(question: str) -> str:
    """Prefix the question with the configured apiary region (reads the apiary_region config var)."""
    return f'[{APIARY_REGION.value()}] {question}'


@pxt.udf
def format_context(hits: pxt.Json) -> str:
    """Turn retrieval hits ([{title, text, score}, ...]) into a compact numbered context block."""
    lines = []
    for i, h in enumerate(hits or [], start=1):
        lines.append(f'{i}. ({h.get("title")}, score={float(h.get("score") or 0):.2f}) {h.get("text")}')
    return '\n'.join(lines)


@pxt.udf
def extractive_answer(question: str, hits: pxt.Json) -> str:
    """Offline answer: the best-overlapping retrieved sentence plus its source title."""
    if not hits:
        return 'No guidance found in the apiary handbook.'
    q = set(_tokens(question))
    best = max(hits, key=lambda h: (len(q & set(_tokens(h.get('text', '')))), float(h.get('score') or 0)))
    return f'{best.get("text")} (source: {best.get("title")})'


@pxt.udf
def advisor_messages(question: str, context: str) -> pxt.Json:
    """Chat messages for the tool-calling advisor."""
    return [
        {
            'role': 'system',
            'content': 'You are an apiary advisor. Use the tools when a mite count or handbook lookup helps. '
            'Answer in at most two sentences.',
        },
        {'role': 'user', 'content': f'Handbook context:\n{context}\n\nQuestion: {question}'},
    ]


@pxt.udf
def final_messages(question: str, context: str, tool_output: pxt.Json) -> pxt.Json:
    """Second-pass messages: the question plus whatever the tools returned."""
    return [
        {'role': 'system', 'content': 'You are an apiary advisor. Answer in at most two sentences using the tool results.'},
        {
            'role': 'user',
            'content': f'Handbook context:\n{context}\n\nTool results: {tool_output}\n\nQuestion: {question}',
        },
    ]
