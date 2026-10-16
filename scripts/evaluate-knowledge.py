"""Local retrieval calibration and 10k synthetic index timing; never call a chat API."""
import argparse
import json
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np

from echomind.knowledge import KnowledgeService
from echomind.knowledge_model import LocalEmbedder, MODEL_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(".runtime/knowledge-evaluation.json"))
    args = parser.parse_args()
    corpus = json.loads(Path(__file__).with_name("knowledge-corpus.json").read_text(encoding="utf-8"))
    embedder = LocalEmbedder(args.model)
    class Cached:
        def __init__(self):
            self.cache = {}
        def encode(self, texts):
            missing = [t for t in texts if t not in self.cache]
            if missing:
                self.cache.update(zip(missing, embedder.encode(missing)))
            return np.array([self.cache[t] for t in texts])
    cached = Cached()
    with tempfile.TemporaryDirectory(prefix="knowledge-eval-") as temporary:
        service = KnowledgeService(temporary, embedder=cached)
        examples = []
        for title, answer, queries in corpus['topics']:
            identifier = service.save_question(title, answer)
            examples.extend((q, identifier) for q in queries)
        examples.extend((q, None) for q in corpus['negative'])
        # Grid calibration is deterministic; the actual model encodes each query once.
        candidates = []
        measurements = []
        for threshold in np.arange(.35, .851, .005):
            service.threshold = float(threshold)
            correct = false = 0
            for question, expected in examples:
                hits = service.retrieve(question)
                correct += expected is not None and expected in {h.document_id for h in hits}
                false += expected is None and bool(hits)
            if correct >= 72 and false <= 1:
                candidates.append((round(float(threshold), 3), correct, false))
            measurements.append((round(float(threshold), 3), correct, false))
        threshold, correct, false = candidates[0] if candidates else max(
            (r for r in measurements if r[2] <= 1), key=lambda r: (r[1], -r[0]))
        service.threshold = threshold
        cases = [{'query':q, 'expected':expected, 'hits':[h.document_id for h in service.retrieve(q)]} for q, expected in examples]
        report = {'model':MODEL_ID, 'threshold':threshold, 'positive':80, 'negative':20,
                  'recall_at_3':correct / 80, 'unrelated_injection_rate':false / 20,
                  'passed':bool(candidates), 'cases':cases, 'calibration':measurements}
        # Repeat the 20 actual embedded chunks to test scan cost at 10k, not to claim 10k-topic accuracy.
        with service.connect() as db:
            original = db.execute('SELECT * FROM chunks').fetchall()
            for i in range(10000 - len(original)):
                row = original[i % len(original)]
                rowid = db.execute('INSERT INTO chunks(document_id,title,location,text,embedding,model) VALUES(?,?,?,?,?,?)',
                                   (row['document_id'], row['title'], row['location'], row['text'], row['embedding'], row['model'])).lastrowid
                db.execute('INSERT INTO search(rowid,title,aliases,tags,text) VALUES(?,?,?,?,?)',
                           (rowid, row['title'], '', '', row['text']))
            db.execute("UPDATE meta SET value=value+1 WHERE key='revision'")
        service._embedder = embedder
        service.retrieve(examples[0][0])
        durations = []
        for question, _ in examples[:30]:
            started = time.perf_counter()
            service.retrieve(question)
            durations.append((time.perf_counter() - started) * 1000)
        report['benchmark'] = {'chunks':10000, 'samples':len(durations), 'p50_ms':statistics.median(durations),
                               'p95_ms':float(np.percentile(durations,95)), 'synthetic_repeated_vectors':True}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('cases','calibration')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
