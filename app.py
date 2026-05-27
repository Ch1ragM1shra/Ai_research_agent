from __future__ import annotations
import os
import re
import time
from datetime import datetime

import streamlit as st

# Disable LangSmith tracing (matches your pipeline)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""

from agents import (  # noqa: E402
    build_search_agent,
    build_search_reader_agent,
    writer_chain,
    critic_chain,
)

# Page config + theme

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
  /* base */
  .stApp {
    background:
      radial-gradient(1200px 600px at 10% -10%, rgba(99,102,241,0.18), transparent 60%),
      radial-gradient(900px 500px at 110% 10%, rgba(236,72,153,0.12), transparent 60%),
      linear-gradient(180deg, #0b0b12 0%, #0a0a10 100%);
    color: #e7e7ee;
  }
  section[data-testid="stSidebar"] {
    background: rgba(15, 15, 22, 0.75);
    backdrop-filter: blur(14px);
    border-right: 1px solid rgba(255,255,255,0.06);
  }

  /* hero */
  .hero {
    padding: 28px 32px;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(236,72,153,0.10) 60%, rgba(34,211,238,0.10));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 20px 60px -20px rgba(99,102,241,0.35);
    margin-bottom: 22px;
  }
  .hero h1 {
    font-size: 2.2rem;
    margin: 0;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #c7d2fe, #f9a8d4 60%, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .hero p { color: #b8b8c7; margin: 6px 0 0; font-size: 0.98rem; }

  /* agent card */
  .agent-card {
    padding: 14px 18px;
    border-radius: 16px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 10px;
    transition: all .25s ease;
  }
  .agent-card.active {
    border-color: rgba(167,139,250,0.55);
    background: rgba(99,102,241,0.10);
    box-shadow: 0 0 0 1px rgba(167,139,250,0.35), 0 10px 30px -10px rgba(99,102,241,0.6);
  }
  .agent-card.done {
    border-color: rgba(74,222,128,0.4);
    background: rgba(34,197,94,0.06);
  }
  .agent-title { font-weight: 600; font-size: 0.98rem; display:flex; align-items:center; gap:10px; }
  .agent-sub   { color:#9aa0b4; font-size: 0.82rem; margin-top: 2px; }
  .pill {
    display:inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: .02em;
  }
  .pill.idle   { background: rgba(255,255,255,0.06); color:#a3a3b2; }
  .pill.active { background: rgba(167,139,250,0.18); color:#c4b5fd; }
  .pill.done   { background: rgba(74,222,128,0.16); color:#86efac; }
  .pill.err    { background: rgba(248,113,113,0.18); color:#fca5a5; }

  /* report container */
  .report-wrap {
    padding: 26px 30px;
    border-radius: 18px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
  }

  /* buttons */
  .stButton>button {
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.10);
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white; font-weight: 600; padding: 0.55rem 1.1rem;
    transition: transform .15s ease, box-shadow .2s ease;
  }
  .stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 30px -10px rgba(139,92,246,0.7);
    border-color: rgba(255,255,255,0.2);
  }

  /* inputs */
  .stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    color: #e7e7ee !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 12px !important;
  }

  /* tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 6px; }
  .stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.03);
    border-radius: 10px 10px 0 0;
    padding: 10px 16px;
  }
  .stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.15) !important;
    color: #c7d2fe !important;
  }

  /* hide default streamlit chrome a bit */
  #MainMenu, footer { visibility: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Helpers
def extract_url(text: str) -> str | None:
    urls = re.findall(r"https?://\S+", text or "")
    return urls[0] if urls else None


AGENTS = [
    ("search",  "  Search Agent",  "Gathers reliable sources on the topic"),
    ("reader",  "  Reader Agent",  "Scrapes the top URL for detailed content"),
    ("writer",  "  Writer Agent",  "Synthesizes the research into a report"),
    ("critic",  "  Critic Agent",  "Reviews the report and gives feedback"),
]

STATUS_LABEL = {"idle": "Idle", "active": "Running", "done": "Done", "err": "Error"}


def render_pipeline_panel(container, status: dict[str, str]):
    html = ['<div>']
    for key, title, sub in AGENTS:
        s = status.get(key, "idle")
        html.append(
            f'<div class="agent-card {s if s in ("active","done") else ""}">'
            f'  <div class="agent-title">{title}'
            f'    <span class="pill {s}">{STATUS_LABEL[s]}</span>'
            f'  </div>'
            f'  <div class="agent-sub">{sub}</div>'
            f'</div>'
        )
    html.append('</div>')
    container.markdown("\n".join(html), unsafe_allow_html=True)



# Session state

ss = st.session_state
ss.setdefault("history", [])      # list of past runs
ss.setdefault("result", None)     # current run result
ss.setdefault("running", False)



# Sidebar

with st.sidebar:
    st.markdown("### Research Agent")
    st.caption("Multi-agent pipeline · Search → Read → Write → Critique")
    st.divider()

    st.markdown("#### Recent runs")
    if not ss.history:
        st.caption("_No runs yet._")
    else:
        for i, h in enumerate(reversed(ss.history[-8:])):
            if st.button(f"{h['topic'][:38]}", key=f"hist-{i}", use_container_width=True):
                ss.result = h
                st.rerun()

    st.divider()
    st.caption(f"Session started · {datetime.now().strftime('%H:%M')}")



# Hero
st.markdown(
    """
    <div class="hero">
      <h1>AI Research Agent</h1>
      <p>Give it any topic. Four agents collaborate to produce a researched, critiqued report.</p>
    </div>
    """,
    unsafe_allow_html=True,
)



# Input

col_input, col_button = st.columns([5, 1])
with col_input:
    topic = st.text_input(
        "Topic",
        placeholder="e.g. The impact of small language models on edge devices in 2025",
        label_visibility="collapsed",
        key="topic_input",
    )
with col_button:
    run_clicked = st.button("Research →", use_container_width=True, disabled=ss.running)



# Pipeline run
left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown("#### Pipeline")
    pipeline_slot = st.empty()
    # initial render
    init_status = {k: "idle" for k, *_ in AGENTS}
    render_pipeline_panel(pipeline_slot, init_status)

with right:
    output_slot = st.container()


def run_pipeline_ui(topic: str):
    status = {k: "idle" for k, *_ in AGENTS}
    state: dict = {"topic": topic, "ts": datetime.now().isoformat(timespec="seconds")}

    def update(stage, value):
        status[stage] = value
        render_pipeline_panel(pipeline_slot, status)

    with output_slot:
        log = st.status("Starting pipeline…", expanded=True)

        # SEARCH
        try:
            update("search", "active")
            log.update(label="Search agent is gathering sources…")
            search_agent = build_search_agent()
            search_results = search_agent.invoke({
                "messages": [("user", f"Find reliable and detailed information about: {topic}")]
            })
            state["search_results"] = search_results["messages"][-1].content
            update("search", "done")
        except Exception as e:
            update("search", "err")
            log.update(label="Search agent failed", state="error")
            st.exception(e); return

        # READER
        url = extract_url(state["search_results"])
        state["url"] = url
        try:
            if not url:
                state["scraped_content"] = "No webpage could be scraped."
                update("reader", "done")
            else:
                update("reader", "active")
                log.update(label=f"Reader agent scraping {url[:60]}…")
                reader_agent = build_search_reader_agent()
                reader_result = reader_agent.invoke({
                    "messages": [(
                        "user",
                        f"Scrape the following webpage URL and extract important detailed "
                        f"information.\n\nURL:\n{url}\n\nOnly use the scrape_web tool."
                    )]
                })
                state["scraped_content"] = reader_result["messages"][-1].content
                update("reader", "done")
        except Exception as e:
            update("reader", "err")
            log.update(label="Reader agent failed", state="error")
            st.exception(e); return

        # WRITER
        try:
            update("writer", "active")
            log.update(label="Writer agent drafting the report…")
            research_combined = (
                f"Search Results:\n{state['search_results']}\n\n"
                f"Detailed Scraped Content:\n{state['scraped_content']}"
            )
            state["report"] = writer_chain.invoke({"topic": topic, "research": research_combined})
            update("writer", "done")
        except Exception as e:
            update("writer", "err")
            log.update(label="Writer agent failed", state="error")
            st.exception(e); return

        # CRITIC
        try:
            update("critic", "active")
            log.update(label="  Critic agent reviewing…")
            state["feedback"] = critic_chain.invoke({"report": state["report"]})
            update("critic", "done")
        except Exception as e:
            update("critic", "err")
            log.update(label="Critic agent failed", state="error")
            st.exception(e); return

        log.update(label="Pipeline complete", state="complete", expanded=False)
        time.sleep(0.2)

    return state


def _to_text(x) -> str:
    """LangChain chains may return strings or message-like objects."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return getattr(x, "content", None) or str(x)


# Trigger run
if run_clicked:
    if not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        ss.running = True
        try:
            result = run_pipeline_ui(topic.strip())
            if result:
                ss.result = result
                ss.history.append(result)
        finally:
            ss.running = False
            st.rerun()



# Results
if ss.result and not ss.running:
    r = ss.result
    with output_slot:
        st.markdown(f"###   Report — *{r.get('topic','')}*")

        tab_report, tab_critic, tab_search, tab_scrape = st.tabs(
            ["Report", "Critic", "Search results", "Scraped content"]
        )

        with tab_report:
            st.markdown('<div class="report-wrap">', unsafe_allow_html=True)
            st.markdown(_to_text(r.get("report")) or "_No report generated._")
            st.markdown("</div>", unsafe_allow_html=True)

            st.download_button(
                " Download report (.md)",
                data=_to_text(r.get("report")),
                file_name=f"report-{re.sub(r'[^a-z0-9]+','-', r.get('topic','').lower())[:60]}.md",
                mime="text/markdown",
            )

        with tab_critic:
            st.markdown(_to_text(r.get("feedback")) or "_No feedback._")

        with tab_search:
            if r.get("url"):
                st.caption(f"Top URL: {r['url']}")
            st.markdown(_to_text(r.get("search_results")) or "_No search output._")

        with tab_scrape:
            st.markdown(_to_text(r.get("scraped_content")) or "_No scraped content._")
