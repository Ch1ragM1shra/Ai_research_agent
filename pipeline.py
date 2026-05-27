import os
import re

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""

from agents import (
    build_search_agent,
    build_search_reader_agent,
    writer_chain,
    critic_chain
)

from rich import print


def extract_url(text: str):

    urls = re.findall(r'https?://\S+', text)

    if urls:
        return urls[0]

    return None


def run_pipeline(topic: str) -> dict:

    state = {}

    # SEARCH AGENT

    print("\n" + "=" * 20)
    print("SEARCH AGENT WORKING")
    print("=" * 20)

    search_agent = build_search_agent()

    search_results = search_agent.invoke({

        "messages": [
            (
                "user",
                f"Find reliable and detailed information about: {topic}"
            )
        ]

    })

    state["search_results"] = search_results["messages"][-1].content

    print("\nSEARCH RESULTS:\n")
    print(state["search_results"])

 

    # EXTRACT URL

    url = extract_url(state["search_results"])

    if not url:

        print("\nNo URL found in search results.\n")

        state["scraped_content"] = "No webpage could be scraped."

    else:

        print("\nExtracted URL:\n")
        print(url)


        # READER AGENT

        print("\n" + "=" * 20)
        print("READER AGENT SCRAPING WEBPAGE")
        print("=" * 20)

        reader_agent = build_search_reader_agent()

        reader_result = reader_agent.invoke({

            "messages": [
                (
                    "user",
                    f"""
Scrape the following webpage URL and extract
important detailed information.

URL:
{url}

Only use the scrape_web tool.
"""
                )
            ]

        })

        state["scraped_content"] = reader_result["messages"][-1].content

        print("\nSCRAPED CONTENT:\n")
        print(state["scraped_content"])

 # COMBINE RESEARCH


    research_combined = f"""
Search Results:
{state['search_results']}

Detailed Scraped Content:
{state['scraped_content']}
"""

    # WRITER AGENT

    print("\n" + "=" * 20)
    print("WRITER AGENT WORKING")
    print("=" * 20)

    state["report"] = writer_chain.invoke({

        "topic": topic,
        "research": research_combined

    })

    print("\nFINAL REPORT:\n")
    print(state["report"])

  
    # CRITIC AGENT

    print("\n" + "=" * 20)
    print("CRITIC AGENT REVIEWING")
    print("=" * 20)

    state["feedback"] = critic_chain.invoke({

        "report": state["report"]

    })

    print("\nCRITIC REPORT:\n")
    print(state["feedback"])

    return state


if __name__ == "__main__":

    topic = input("\nEnter research topic: ")

    run_pipeline(topic)