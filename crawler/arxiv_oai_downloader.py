# crawler/arxiv_oai_downloader.py
import requests
import xml.etree.ElementTree as ET
import json
import time
from tqdm import tqdm

def fetch_records(set_name, max_records=25000):
    base_url = "https://export.arxiv.org/oai2"
    records = []
    url = f"{base_url}?verb=ListRecords&set={set_name}&metadataPrefix=arXiv"
    resumption_token = None
    total_fetched = 0

    with tqdm(desc=f"Fetching {set_name}", unit="records") as pbar:
        while total_fetched < max_records:
            params = {"verb": "ListRecords", "set": set_name, "metadataPrefix": "arXiv"}
            if resumption_token:
                params = {"verb": "ListRecords", "resumptionToken": resumption_token}
            
            try:
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                root = ET.fromstring(response.content)
                
                # Parse records
                for record in root.findall(".//{http://www.openarchives.org/OAI/2.0/}record"):
                    header = record.find(".//{http://www.openarchives.org/OAI/2.0/}header")
                    metadata = record.find(".//{http://www.openarchives.org/OAI/2.0/}metadata")
                    if header is None or metadata is None:
                        continue
                    
                    identifier = header.find("{http://www.openarchives.org/OAI/2.0/}identifier").text
                    arxiv_id = identifier.split(":")[-1]  # e.g., 'cs/9711001'
                    
                    # Parse arXiv metadata
                    arxiv_meta = metadata.find("{http://arxiv.org/OAI/arXiv/}arXiv")
                    if arxiv_meta is None:
                        continue
                    
                    title = arxiv_meta.find("{http://arxiv.org/OAI/arXiv/}title").text if arxiv_meta.find("{http://arxiv.org/OAI/arXiv/}title") is not None else "No title"
                    authors = [a.text for a in arxiv_meta.findall("{http://arxiv.org/OAI/arXiv/}authors/{http://arxiv.org/OAI/arXiv/}author")]
                    abstract = arxiv_meta.find("{http://arxiv.org/OAI/arXiv/}abstract").text if arxiv_meta.find("{http://arxiv.org/OAI/arXiv/}abstract") is not None else "No abstract"
                    categories = arxiv_meta.find("{http://arxiv.org/OAI/arXiv/}categories").text if arxiv_meta.find("{http://arxiv.org/OAI/arXiv/}categories") is not None else "unknown"
                    
                    records.append({
                        "id": arxiv_id,
                        "title": title,
                        "authors": authors,
                        "abstract": abstract,
                        "categories": categories,
                        "url": f"https://arxiv.org/abs/{arxiv_id}"
                    })
                    
                    total_fetched += 1
                    pbar.update(1)
                    if total_fetched >= max_records:
                        break
                
                # Check for resumptionToken
                resumption = root.find(".//{http://www.openarchives.org/OAI/2.0/}resumptionToken")
                if resumption is None or resumption.text is None:
                    break
                resumption_token = resumption.text
                
                # Polite delay
                time.sleep(3)  # Respect rate limits (3s/request)
                
            except Exception as e:
                print(f"Error fetching {set_name}: {e}")
                break
    
    return records

# Download from both sets
print("Downloading CS records...")
cs_records = fetch_records("cs", 25000)

print("Downloading q-bio records...")
qbio_records = fetch_records("q-bio", 25000)

# Combine and save as JSONL
all_records = cs_records + qbio_records
output_path = "data/arxiv_docs.jsonl"

with open(output_path, "w") as f:
    for record in all_records:
        f.write(json.dumps(record) + "\n")

print(f"Saved {len(all_records)} records to {output_path}")