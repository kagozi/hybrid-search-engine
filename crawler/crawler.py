# crawler/crawler.py
import aiohttp
import asyncio
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import json
from collections import deque

class PoliteCrawler:
    def __init__(self, start_urls, max_pages=50000, delay=0.6):
        self.start_urls = start_urls
        self.max_pages = max_pages
        self.delay = delay
        self.visited = set()
        self.queue = deque()
        self.documents = []

    async def fetch(self, session, url):
        try:
            async with session.get(url, timeout=10) as resp:
                return await resp.text()
        except:
            return None

    async def parse(self, session, url):
        if url in self.visited or len(self.documents) >= self.max_pages:
            return
        self.visited.add(url)
        html = await self.fetch(session, url)
        if not html:
            return

        soup = BeautifulSoup(html, "html.parser")
        title = soup.find("title").get_text(strip=True) if soup.find("title") else "No title"
        text = " ".join(p.get_text() for p in soup.find_all("p"))

        if len(text.split()) > 50:
            self.documents.append({
                "url": url,
                "title": title,
                "text": text[:10000],
                "domain": urlparse(url).netloc
            })

        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"])
            if link.startswith("http") and link not in self.visited:
                self.queue.append(link)

        await asyncio.sleep(self.delay)

    async def run(self):
        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            self.queue.extend(self.start_urls)
            while self.queue and len(self.documents) < self.max_pages:
                url = self.queue.popleft()
                await self.parse(session, url)
                if len(self.documents) % 1000 == 0:
                    print(f"Crawled {len(self.documents)} docs...")

        with open("../data/arxiv_docs.jsonl", "w") as f:
            for d in self.documents:
                f.write(json.dumps(d) + "\n")
        print(f"Saved {len(self.documents)} documents.")

if __name__ == "__main__":
    seeds = [
        "https://arxiv.org/list/cs/recent",
        "https://arxiv.org/list/q-bio/recent"
    ]
    crawler = PoliteCrawler(seeds, max_pages=50000)
    asyncio.run(crawler.run())