#!/usr/bin/env python3
"""Apiary Advisor driver (rag-agent shape, no HTTP router).

1. pxt schema update app.py <catalog>   (tables, chunk view, custom embedding index)
2. Load guides -> sentence chunks are embedded by the custom hash_embed UDF
3. Ask the Advisor agent table (retrieval + context + extractive answer)
4. Turn the Advisor table into a function with pxt.udf(table, return_value=...) and apply it to inspections
5. Inspections: mite load via an expr_udf, varroa action via a UDF
6. guides_by_topic: functions.net.presigned_url links for object-storage sources
7. Questions: tiny LLM tool-calling pass (gpt-4o-mini; skipped without OPENAI_API_KEY)
8. Config ops: show where the apiary_region var resolves from
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pixeltable as pxt

ROOT = Path(__file__).resolve().parent
CATALOG = os.environ.get('APIARY_CATALOG', 'apiary')
PXT = str(Path(sys.executable).with_name('pxt'))
LOG = ROOT / 'logs' / 'run.jsonl'


def log(event: str, **payload) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a') as f:
        f.write(json.dumps({'event': event, **payload}, default=str) + '\n')
    print(f'{event}: {json.dumps(payload, default=str)[:600]}', flush=True)


def pxt_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([PXT, *args], cwd=str(ROOT), capture_output=True, text=True)


def rows(rs) -> list[dict]:
    return [dict(r) for r in rs]


def main() -> int:
    if LOG.exists():
        LOG.unlink()

    t0 = time.time()
    r = pxt_cli('schema', 'update', 'app.py', CATALOG, '-f')
    log('schema_update', exit=r.returncode, seconds=round(time.time() - t0, 2), out=r.stdout.strip()[-400:], err=r.stderr.strip()[-400:])
    if r.returncode != 0:
        return 1

    guides = pxt.get_table(f'{CATALOG}/guides')
    chunks = pxt.get_table(f'{CATALOG}/guide_chunks')
    advisor = pxt.get_table(f'{CATALOG}/advisor')
    insp = pxt.get_table(f'{CATALOG}/inspections')
    questions = pxt.get_table(f'{CATALOG}/questions')

    # --- 2. guides -> chunks -> custom embedding index ---
    data = json.loads((ROOT / 'data' / 'guides.json').read_text())
    t0 = time.time()
    st = guides.insert(data, on_error='abort') if guides.count() == 0 else None
    log('guides_loaded', guides=guides.count(), chunks=chunks.count(), seconds=round(time.time() - t0, 2), status=str(st))

    q = 'how many mites per 100 bees means I should treat'
    sim = chunks.text.similarity(string=q, idx='chunk_embed')
    top = rows(chunks.order_by(sim, asc=False).limit(3).select(chunks.title, chunks.text, score=sim).collect())
    log('similarity_top3', question=q, top=top)

    # --- 3. Advisor agent table ---
    advisor.delete()
    asks = [
        'how many mites per 100 bees means I should treat',
        'what should I do about swarm cells in spring',
        'how do I know if my colony is queenless',
    ]
    t0 = time.time()
    advisor.insert([{'question': a} for a in asks])
    adv = rows(advisor.select(advisor.question, advisor.regional_question, advisor.answer).collect())
    log('advisor_answers', seconds=round(time.time() - t0, 2), n=len(adv), sample=adv)

    # --- 5. inspections (expr_udf + udf) ---
    insp.delete()
    t0 = time.time()
    insp.insert(json.loads((ROOT / 'data' / 'inspections.json').read_text()))
    loads = rows(insp.select(insp.hive_id, insp.mites, insp.bees_sampled, insp.mite_pct, insp.action).order_by(insp.hive_id).collect())
    log('inspections', seconds=round(time.time() - t0, 2), n=len(loads), rows=loads)
    hist = rows(insp.where(insp.hive_id == 'H-ridge-2').select(insp.mites, insp.mite_pct, insp.action).collect())
    log('hive_history_ridge2', rows=hist)

    # --- 4. table-as-UDF: the Advisor pipeline as a reusable function ---
    advise = pxt.udf(advisor, return_value=advisor.answer)
    before = advisor.count()
    t0 = time.time()
    advised = rows(insp.select(insp.hive_id, insp.note, advice=advise(question=insp.note)).order_by(insp.hive_id).collect())
    log('table_udf_advice', seconds=round(time.time() - t0, 2), n=len(advised), advisor_rows_before=before, advisor_rows_after=advisor.count(), rows=advised)
    full = pxt.udf(advisor)  # no return_value: a dict of every column of the pipeline
    one = rows(insp.where(insp.hive_id == 'H-meadow-3').select(out=full(question=insp.note)).collect())
    log('table_udf_dict_keys', keys=sorted(one[0]['out'].keys()) if one else [])

    # --- 6. functions.net.presigned_url via @pxt.query-equivalent filter ---
    links = rows(
        guides.select(guides.guide_id, guides.source_uri, link=pxt.functions.net.presigned_url(guides.source_uri, 3600))
        .order_by(guides.guide_id)
        .collect()
    )
    log('presigned_links', rows=links)

    # --- 7. LLM tool calling (tiny) ---
    if os.environ.get('OPENAI_API_KEY'):
        questions.delete()
        t0 = time.time()
        st = questions.insert(
            [
                {'question': 'My wash found 9 mites in 300 bees. What should I do? Use the varroa_action tool with mites per 100 bees.'},
                {'question': 'How should I feed a light hive before winter?'},
            ],
            on_error='ignore',
        )
        res = rows(questions.select(questions.question, questions.tool_output, questions.answer).collect())
        errs = rows(
            questions.where(questions.response.errortype != None)  # noqa: E711
            .select(err=questions.response.errortype, msg=questions.response.errormsg)
            .collect()
        )
        log('llm_tool_calls', seconds=round(time.time() - t0, 2), status=str(st), rows=res, errors=errs)
    else:
        log('llm_tool_calls', skipped='OPENAI_API_KEY not set')

    # --- 8. config ops ---
    r = pxt_cli('config', '--section', 'pixeltable.database.vars')
    log('config_vars', out=r.stdout.strip())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
