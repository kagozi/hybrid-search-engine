"""
Quick script to check corpus file format
"""
import json

datasets = ["scifact", "trec-covid"]

for dataset in datasets:
    print(f"\n{'='*60}")
    print(f"Checking {dataset.upper()}")
    print(f"{'='*60}")
    
    with open(f"data/beir/{dataset}/corpus.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 3:  # Check first 3 documents
                break
            
            doc = json.loads(line)
            print(f"\nDocument {i+1}:")
            print(f"  Keys: {list(doc.keys())}")
            print(f"  _id: {doc.get('_id', 'N/A')}")
            print(f"  id: {doc.get('id', 'N/A')}")
            print(f"  title: {doc.get('title', 'N/A')[:50]}...")
            print(f"  text length: {len(doc.get('text', ''))}")