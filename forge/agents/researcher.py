"""Researcher agent: web search, information synthesis, and citation tracking."""

from __future__ import annotations

from forge.agents.base import BaseAgent, AgentConfig
from forge.llm.client import OllamaClient

RESEARCHER_SYSTEM_PROMPT = """\
You are a research assistant running locally via Ollama. Your capabilities:

1. **Web search**: Use web_search to find current information
2. **Fetch pages**: Use web_fetch to read full web pages
3. **Read files**: Use read_file to examine local documents
4. **Write files**: Use write_file to save research findings

Guidelines:
- Always search the web before answering questions about current events or recent topics.
- Cite your sources — include URLs and titles.
- Synthesize information from multiple sources when possible.
- Be transparent about uncertainty — if results are conflicting, say so.
- Organize findings clearly with headings and bullet points.
- Save important research to files when the user asks.
- Distinguish between facts, widely-held opinions, and speculation.
"""


def create_researcher_agent(client: OllamaClient, working_dir: str = ".") -> BaseAgent:
    """Create a researcher agent with optimal settings for research tasks."""
    config = AgentConfig(
        name="researcher",
        system_prompt=RESEARCHER_SYSTEM_PROMPT,
        tools=["web", "filesystem"],
        temperature=0.5,
        description="Research assistant — searches web, summarizes findings",
    )
    return BaseAgent(client=client, config=config, working_dir=working_dir)
