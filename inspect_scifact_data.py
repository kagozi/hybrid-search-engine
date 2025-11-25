"""Inspect SciFact queries and qrels to understand the mismatch"""
import json

print("="*60)
print("SciFact Queries (first 10)")
print("="*60)
with open("data/beir/scifact/queries.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        q = json.loads(line)
        print(f"ID: {q['_id']:>4} | Text: {q['text'][:60]}...")

print("\n" + "="*60)
print("SciFact Qrels (first 10)")
print("="*60)
with open("data/beir/scifact/qrels/test.tsv") as f:
    for i, line in enumerate(f):
        if i == 0:
            print(f"Header: {line.strip()}")
            continue
        if i > 10:
            break
        parts = line.strip().split('\t')
        print(f"Query ID: {parts[0]:>4} | Doc ID: {parts[2]:>6} | Score: {parts[3] if len(parts) > 3 else '1'}")

print("\n" + "="*60)
print("Checking for overlap")
print("="*60)

# Load all query IDs
with open("data/beir/scifact/queries.jsonl") as f:
    query_ids = set(str(json.loads(line)['_id']) for line in f)

# Load all qrel query IDs
with open("data/beir/scifact/qrels/test.tsv") as f:
    lines = f.readlines()
    qrel_ids = set(line.strip().split('\t')[0] for line in lines[1:])  # Skip header

print(f"Total queries: {len(query_ids)}")
print(f"Total qrels: {len(qrel_ids)}")
print(f"Overlap: {len(query_ids & qrel_ids)}")
print(f"\nFirst 10 query IDs: {sorted(list(query_ids))[:10]}")
print(f"First 10 qrel IDs: {sorted(list(qrel_ids))[:10]}")