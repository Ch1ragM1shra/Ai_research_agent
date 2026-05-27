
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from langchain.tools import tool
import os
from dotenv import load_dotenv
load_dotenv(".env")


from rich import print

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
from langchain.tools import tool
from tavily import TavilyClient
import os

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information.

    Args:
        query: The search query to search on the internet.
    """

    results = tavily.search(
        query=query,
        max_results=3
    )

    out = []

    for r in results["results"]:

        out.append(
            f"Title: {r['title']}\n"
            f"Content: {r['content']}\n"
            f"URL: {r['url']}\n"
        )

    return "\n\n".join(out)


@tool
def scrape_web(url : str)->str:
    """Scrape detailed content from a webpage URL."""
    
    try:
     response = requests.get(url, timeout=8, headers ={"User-Agent": "Mozilla/5.0"})
     soup = BeautifulSoup(response.text, 'html.parser')
     for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
     return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        return f"Error scraping the web: {str(e)}"








