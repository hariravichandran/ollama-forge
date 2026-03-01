"""ollama-forge CLI — main entry point for all commands."""

from __future__ import annotations

import os
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """ollama-forge: Batteries-included local AI framework for Ollama.

    Run 'forge' with no arguments to start an interactive chat.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(chat)


# ─── Chat ────────────────────────────────────────────────────────────────────


@main.command()
@click.option("--model", "-m", default="", help="Model to use (auto-detected if empty)")
@click.option("--agent", "-a", default="assistant", help="Agent to use (assistant, coder, researcher)")
@click.option("--working-dir", "-d", default=".", help="Working directory for file operations")
@click.option("--cascade", is_flag=True, help="Enable cascading: auto-switch to bigger model when stuck")
@click.option("--auto-approve", is_flag=True, help="Auto-approve all tool actions (skip permission prompts)")
@click.option("--image", "-i", multiple=True, help="Image file(s) to include (for vision models)")
def chat(model: str, agent: str, working_dir: str, cascade: bool, auto_approve: bool, image: tuple):
    """Start an interactive chat session."""
    from forge.config import load_config
    from forge.hardware import detect_hardware, select_profile
    from forge.hardware.rocm import configure_rocm_env
    from forge.llm.client import OllamaClient
    from forge.agents.orchestrator import AgentOrchestrator
    from forge.agents.permissions import PermissionManager, AutoApproveManager
    from forge.agents.memory import ConversationMemory

    config = load_config()

    # Detect hardware and configure
    hw = detect_hardware()
    profile = select_profile(hw)
    configure_rocm_env(hw.gpu)

    # Select model
    model_name = model or config.default_model or profile.recommended_model

    # Initialize client
    client = OllamaClient(
        model=model_name,
        base_url=config.ollama_base_url,
        num_ctx=profile.num_ctx,
        num_thread=profile.max_threads,
        num_batch=profile.num_batch,
    )

    if not client.is_available():
        console.print("[red]Ollama is not running.[/red] Start it with: ollama serve")
        sys.exit(1)

    # Permission manager
    permissions = AutoApproveManager() if auto_approve else PermissionManager()

    # Conversation memory
    memory = ConversationMemory()
    facts_context = memory.get_facts_context()

    # Initialize orchestrator
    orchestrator = AgentOrchestrator(client=client, working_dir=working_dir)
    if agent != "assistant":
        orchestrator.switch_agent(agent)

    # Enable cascade mode if requested
    if cascade:
        from forge.agents.cascade import auto_cascade_config, CascadeAgent, CascadeConfig
        from forge.agents.base import AgentConfig
        gpu_gb = hw.gpu.usable_gb if hw.gpu else 0
        cc = auto_cascade_config(gpu_gb)
        if cc.escalation_model:
            console.print(f"[dim]Cascade: {cc.primary_model} → {cc.escalation_model}[/dim]")
        else:
            console.print("[dim]Cascade: no escalation model available for your hardware[/dim]")

    # Print welcome
    memory_status = f"  Memory: {len(memory._facts)} facts" if memory._facts else ""
    console.print(Panel(
        f"[bold]ollama-forge[/bold] v0.1.0\n"
        f"Model: [cyan]{model_name}[/cyan]  Agent: [green]{orchestrator.active_agent}[/green]\n"
        f"Hardware: {profile.name} ({hw.gpu.name}){memory_status}\n\n"
        f"Commands: /agents, /agent <name>, /model <name>, /reset, /stats, /idea,\n"
        f"          /remember <fact>, /forget, /quit",
        title="Welcome",
        border_style="blue",
    ))

    # Chat loop
    while True:
        try:
            user_input = console.input("\n[bold blue]You>[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ("/quit", "/exit", "exit", "quit"):
            # Save conversation before exiting
            current_agent = orchestrator.agents.get(orchestrator.active_agent)
            if current_agent and current_agent.messages:
                memory.save_conversation(current_agent.messages)
            console.print("Goodbye!")
            break
        elif user_input == "/reset":
            current = orchestrator.agents.get(orchestrator.active_agent)
            if current:
                current.reset()
            console.print("[dim]Conversation reset.[/dim]")
            continue
        elif user_input == "/stats":
            stats = orchestrator.get_all_stats()
            console.print_json(data=stats)
            continue
        elif user_input.startswith("/model "):
            new_model = user_input[7:].strip()
            if client.switch_model(new_model):
                console.print(f"Switched to model: [cyan]{new_model}[/cyan]")
            else:
                console.print(f"[red]Failed to switch to {new_model}[/red]")
            continue
        elif user_input.startswith("/remember "):
            fact = user_input[10:].strip()
            if ":" in fact:
                key, value = fact.split(":", 1)
                memory.store_fact(key.strip(), value.strip())
                console.print(f"[green]Remembered: {key.strip()} = {value.strip()}[/green]")
            else:
                memory.store_fact(f"note_{len(memory._facts)}", fact)
                console.print(f"[green]Remembered: {fact}[/green]")
            continue
        elif user_input == "/forget":
            memory.clear()
            console.print("[dim]Memory cleared.[/dim]")
            continue
        elif user_input.startswith("/idea"):
            _handle_idea_command(user_input, working_dir)
            continue
        elif user_input.startswith("/"):
            # Pass to orchestrator (handles /agents, /agent <name>)
            response = orchestrator.chat(user_input)
            console.print(f"\n[dim]{response}[/dim]")
            continue

        # Normal message
        with console.status("[bold green]Thinking...", spinner="dots"):
            response = orchestrator.chat(user_input)

        console.print(f"\n[bold green]Agent>[/bold green] {response}")


def _handle_idea_command(user_input: str, working_dir: str):
    """Handle /idea commands in chat."""
    from forge.community.ideas import IdeaCollector

    collector = IdeaCollector(ideas_dir=os.path.join(working_dir, "community"))

    parts = user_input.split(maxsplit=1)
    if len(parts) == 1 or parts[1].strip() == "list":
        console.print(collector.format_ideas())
    elif parts[1].strip().startswith("submit"):
        text = parts[1].replace("submit", "", 1).strip()
        if not text:
            console.print("Usage: /idea submit <your idea description>")
        else:
            result = collector.submit(title=text[:80], description=text)
            console.print(f"[green]{result}[/green]")
    else:
        console.print("Usage: /idea [list|submit <description>]")


# ─── Hardware ────────────────────────────────────────────────────────────────


@main.command()
def hardware():
    """Show detected hardware and recommended profile."""
    from forge.hardware import detect_hardware, select_profile
    from forge.hardware.rocm import get_rocm_status, configure_rocm_env

    hw = detect_hardware()
    profile = select_profile(hw)
    configure_rocm_env(hw.gpu)
    rocm_status = get_rocm_status()

    console.print(Panel(hw.summary(), title="Detected Hardware", border_style="blue"))

    table = Table(title="Hardware Profile")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Profile", profile.name)
    table.add_row("Recommended model", profile.recommended_model)
    table.add_row("Fallback model", profile.fallback_model)
    table.add_row("Larger model", profile.larger_model)
    table.add_row("Context window", f"{profile.num_ctx:,}")
    table.add_row("Batch size", str(profile.num_batch))
    table.add_row("Max threads", str(profile.max_threads))
    console.print(table)

    if rocm_status:
        table2 = Table(title="ROCm Status")
        table2.add_column("Key", style="cyan")
        table2.add_column("Value", style="green")
        for k, v in rocm_status.items():
            table2.add_row(k, v)
        console.print(table2)


# ─── Models ──────────────────────────────────────────────────────────────────


@main.group(invoke_without_command=True)
@click.pass_context
def models(ctx):
    """Manage Ollama models — list, pull, recommend, auto-update."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(models_list)


@models.command("list")
def models_list():
    """List locally available models."""
    from forge.llm.client import OllamaClient
    from forge.llm.models import estimate_model_size

    client = OllamaClient()
    if not client.is_available():
        console.print("[red]Ollama is not running.[/red]")
        return

    local_models = client.list_models()
    if not local_models:
        console.print("No models installed. Use 'forge models pull <name>' or 'forge models recommend'.")
        return

    table = Table(title="Local Models")
    table.add_column("Name", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Modified", style="dim")
    for m in local_models:
        name = m.get("name", "")
        size_bytes = m.get("size", 0)
        size_gb = size_bytes / (1024 ** 3)
        modified = m.get("modified_at", "")[:10]
        table.add_row(name, f"{size_gb:.1f} GB", modified)
    console.print(table)

    # Show running models
    running = client.list_running()
    if running:
        console.print("\n[bold]Currently loaded:[/bold]")
        for m in running:
            console.print(f"  {m.get('name', '')} ({m.get('size', 0) / 1e9:.1f} GB)")


@models.command("pull")
@click.argument("model_name")
def models_pull(model_name: str):
    """Pull a model from the Ollama registry."""
    from forge.llm.client import OllamaClient

    client = OllamaClient()
    if not client.is_available():
        console.print("[red]Ollama is not running.[/red]")
        return

    with console.status(f"Pulling {model_name}...", spinner="dots"):
        def progress(status):
            console.print(f"  {status}", end="\r")

        success = client.pull_model(model_name, progress_cb=progress)

    if success:
        console.print(f"[green]Successfully pulled {model_name}[/green]")
    else:
        console.print(f"[red]Failed to pull {model_name}[/red]")


@models.command("remove")
@click.argument("model_name")
def models_remove(model_name: str):
    """Remove a locally stored model."""
    from forge.llm.client import OllamaClient

    client = OllamaClient()
    if client.delete_model(model_name):
        console.print(f"[green]Removed {model_name}[/green]")
    else:
        console.print(f"[red]Failed to remove {model_name}[/red]")


@models.command("recommend")
def models_recommend():
    """Recommend models based on your hardware."""
    from forge.hardware import detect_hardware
    from forge.hardware.profiles import recommend_models

    hw = detect_hardware()
    recs = recommend_models(hw)

    table = Table(title="Recommended Models")
    table.add_column("Category", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Reason", style="dim")
    for r in recs:
        table.add_row(r["category"], r["model"], r["reason"])
    console.print(table)
    console.print("\nPull a model with: [bold]forge models pull <model>[/bold]")


@models.command("auto-update")
def models_auto_update():
    """Check for and pull latest versions of installed models."""
    from forge.llm.client import OllamaClient

    client = OllamaClient()
    if not client.is_available():
        console.print("[red]Ollama is not running.[/red]")
        return

    local_models = client.list_models()
    if not local_models:
        console.print("No models installed.")
        return

    console.print(f"Checking updates for {len(local_models)} models...\n")
    for m in local_models:
        name = m.get("name", "")
        console.print(f"  Updating [cyan]{name}[/cyan]...", end=" ")
        success = client.pull_model(name)
        if success:
            console.print("[green]up to date[/green]")
        else:
            console.print("[yellow]failed[/yellow]")


# ─── MCP ─────────────────────────────────────────────────────────────────────


@main.group(invoke_without_command=True)
@click.pass_context
def mcp(ctx):
    """Manage MCP (Model Context Protocol) servers."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(mcp_list)


@mcp.command("list")
def mcp_list():
    """List available and enabled MCP servers."""
    from forge.mcp.manager import MCPManager

    manager = MCPManager()
    available = manager.list_available()

    table = Table(title="MCP Servers")
    table.add_column("Status", style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="dim")
    table.add_column("Description")

    for m in available:
        status_style = {
            "enabled": "[green]ON[/green]",
            "built-in": "[green]built-in[/green]",
            "disabled": "[red]OFF[/red]",
            "available": "[dim]available[/dim]",
        }.get(m["status"], m["status"])
        table.add_row(status_style, m["name"], m["category"], m["description"])
    console.print(table)


@mcp.command("add")
@click.argument("name")
def mcp_add(name: str):
    """Enable an MCP server."""
    from forge.mcp.manager import MCPManager

    manager = MCPManager()
    result = manager.enable(name)
    console.print(result)


@mcp.command("remove")
@click.argument("name")
def mcp_remove(name: str):
    """Disable an MCP server."""
    from forge.mcp.manager import MCPManager

    manager = MCPManager()
    result = manager.disable(name)
    console.print(result)


@mcp.command("search")
@click.argument("query")
def mcp_search(query: str):
    """Search the MCP registry."""
    from forge.mcp.registry import search_registry

    results = search_registry(query)
    if not results:
        console.print(f"No MCPs found matching '{query}'")
        return

    for r in results:
        console.print(f"  [cyan]{r.name}[/cyan] ({r.category})")
        console.print(f"    {r.description}")
        if r.install_cmd:
            console.print(f"    Install: {r.install_cmd}")
        console.print()


# ─── Agents ──────────────────────────────────────────────────────────────────


@main.group(invoke_without_command=True)
@click.pass_context
def agent(ctx):
    """Manage agents — create, list, run."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(agent_list)


@agent.command("list")
def agent_list():
    """List all registered agents."""
    from forge.agents.tracker import AgentTracker
    from pathlib import Path

    # List agent YAML files
    agents_dir = Path("agents")
    yamls = list(agents_dir.glob("*.yaml")) + list(agents_dir.glob("*.yml")) if agents_dir.exists() else []

    table = Table(title="Agents")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="dim")
    table.add_column("Source")

    # Built-in agents
    for name in ["assistant", "coder", "researcher"]:
        table.add_row(name, "built-in", "forge/agents/")

    # User agents
    for yaml_file in yamls:
        table.add_row(yaml_file.stem, "custom", str(yaml_file))

    console.print(table)

    # Show tracked systems
    tracker = AgentTracker()
    if tracker.systems:
        console.print(f"\n{tracker.list_systems()}")


@agent.command("create")
@click.option("--name", prompt="Agent name", help="Name for the new agent")
@click.option("--description", prompt="Description", help="What does this agent do?")
@click.option("--model", default="", help="Model to use (auto-detected if empty)")
def agent_create(name: str, description: str, model: str):
    """Create a new agent interactively."""
    system_prompt = click.prompt("System prompt", default=f"You are a {description.lower()}. Be helpful and concise.")
    tools_str = click.prompt("Tools (comma-separated)", default="filesystem,shell,web")
    tools = [t.strip() for t in tools_str.split(",")]

    from forge.llm.client import OllamaClient
    from forge.agents.orchestrator import AgentOrchestrator

    client = OllamaClient()
    orchestrator = AgentOrchestrator(client=client)
    result = orchestrator.create_agent(
        name=name,
        description=description,
        system_prompt=system_prompt,
        tools=tools,
        model=model,
    )
    console.print(f"[green]{result}[/green]")


@agent.command("run")
@click.argument("name")
def agent_run(name: str):
    """Start a chat session with a specific agent."""
    from forge.config import load_config
    from forge.hardware import detect_hardware, select_profile
    from forge.llm.client import OllamaClient
    from forge.agents.orchestrator import AgentOrchestrator

    config = load_config()
    hw = detect_hardware()
    profile = select_profile(hw)

    client = OllamaClient(
        model=config.default_model or profile.recommended_model,
        num_ctx=profile.num_ctx,
        num_thread=profile.max_threads,
    )

    orchestrator = AgentOrchestrator(client=client)
    result = orchestrator.switch_agent(name)
    console.print(result)

    # Start chat
    main.invoke(main, ctx=click.Context(chat, info_name="chat"), agent=name)


# ─── Ideas ───────────────────────────────────────────────────────────────────


@main.group(invoke_without_command=True)
@click.pass_context
def idea(ctx):
    """Community ideas — submit, list, and manage improvement suggestions."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(idea_list)


@idea.command("list")
def idea_list():
    """List community ideas."""
    from forge.community.ideas import IdeaCollector

    collector = IdeaCollector()
    console.print(collector.format_ideas())


@idea.command("submit")
@click.argument("description", nargs=-1)
def idea_submit(description: tuple):
    """Submit a new improvement idea."""
    from forge.community.ideas import IdeaCollector

    text = " ".join(description)
    if not text:
        text = click.prompt("Describe your idea")

    category = click.prompt(
        "Category",
        type=click.Choice(["feature", "improvement", "bugfix", "performance", "ux", "other"]),
        default="improvement",
    )

    collector = IdeaCollector()
    result = collector.submit(title=text[:80], description=text, category=category)
    console.print(f"[green]{result}[/green]")


# ─── Self-Improve ────────────────────────────────────────────────────────────


@main.command("self-improve")
@click.option("--iterations", "-n", default=1, help="Number of improvement iterations")
@click.option("--enable", is_flag=True, help="Enable self-improvement (opt-in required on first use)")
@click.option("--maintainer", is_flag=True, help="Run in maintainer mode (direct push to main)")
def self_improve(iterations: int, enable: bool, maintainer: bool):
    """Run the self-improvement agent to iterate on ollama-forge.

    This feature is OPT-IN — disabled by default. Enable it with:

      forge self-improve --enable

    Or set FORGE_SELF_IMPROVE=1 in your .env file.

    \b
    Two modes:
    - Contributor (default): Creates GitHub PRs for review. Requires `gh` CLI.
    - Maintainer (--maintainer): Direct push to main + stable promotion.
    """
    from forge.config import load_config, save_config
    from forge.hardware import detect_hardware, select_profile
    from forge.llm.client import OllamaClient
    from forge.community.ideas import IdeaCollector
    from forge.community.self_improve import SelfImproveAgent

    config = load_config()

    # Handle --enable flag: persist the opt-in
    if enable and not config.self_improve_enabled:
        config.self_improve_enabled = True
        save_config(config)
        console.print("[green]Self-improvement enabled.[/green] Thank you for contributing!")

    # Check opt-in
    if not config.self_improve_enabled:
        console.print(
            "[yellow]Self-improvement is disabled by default.[/yellow]\n\n"
            "This feature uses your spare CPU/GPU resources to improve ollama-forge.\n"
            "Improvements are submitted as GitHub PRs for review.\n\n"
            "To enable, run:\n"
            "  [bold]forge self-improve --enable[/bold]\n\n"
            "Or set [bold]FORGE_SELF_IMPROVE=1[/bold] in your .env file.\n\n"
            "You can disable it anytime with [bold]FORGE_SELF_IMPROVE=0[/bold]."
        )
        return

    # Determine mode
    is_maintainer = maintainer or config.self_improve_maintainer
    mode_label = "maintainer (direct push)" if is_maintainer else "contributor (PRs)"

    hw = detect_hardware()
    profile = select_profile(hw)

    client = OllamaClient(
        model=config.default_model or profile.recommended_model,
        num_ctx=profile.num_ctx,
        num_thread=profile.max_threads,
    )

    if not client.is_available():
        console.print("[red]Ollama is not running.[/red]")
        return

    collector = IdeaCollector()
    agent = SelfImproveAgent(
        client=client,
        idea_collector=collector,
        maintainer=is_maintainer,
    )

    console.print(f"[bold]Self-improvement agent[/bold] — mode: [cyan]{mode_label}[/cyan]\n")

    for i in range(iterations):
        console.print(f"\n[bold]Iteration {i + 1}/{iterations}[/bold]")
        with console.status("Evaluating improvements...", spinner="dots"):
            result = agent.run_iteration()

        if result:
            if result.pr_url:
                status = f"[green]PR created: {result.pr_url}[/green]"
            elif result.committed:
                status = "[green]committed to main[/green]"
            else:
                status = "[yellow]tested only[/yellow]"
            console.print(f"  Result: {result.description} — {status}")
            console.print(f"  Files: {', '.join(result.files_changed)}")
            console.print(f"  Tests: {'passed' if result.tests_passed else 'failed'}")
        else:
            console.print("  [dim]No improvement found this iteration[/dim]")


# ─── Benchmark ───────────────────────────────────────────────────────────────


@main.command()
def benchmark():
    """Run a hardware benchmark for model recommendations."""
    from forge.hardware import detect_hardware, select_profile
    from forge.llm.client import OllamaClient
    import time

    hw = detect_hardware()
    profile = select_profile(hw)

    console.print(Panel(hw.summary(), title="Hardware", border_style="blue"))

    client = OllamaClient(
        model=profile.recommended_model,
        num_ctx=profile.num_ctx,
        num_thread=profile.max_threads,
    )

    if not client.is_available():
        console.print("[red]Ollama is not running.[/red]")
        return

    # Check if recommended model is available
    models = [m.get("name", "") for m in client.list_models()]
    if profile.recommended_model not in models:
        console.print(f"Model {profile.recommended_model} not installed. Pull it first:")
        console.print(f"  forge models pull {profile.recommended_model}")
        return

    console.print(f"\nBenchmarking [cyan]{profile.recommended_model}[/cyan]...\n")

    # Run a few test prompts
    prompts = [
        "Write a Python function to check if a number is prime.",
        "Explain the difference between a list and a tuple in Python.",
        "What is the time complexity of binary search?",
    ]

    total_tokens = 0
    total_time = 0.0

    for prompt in prompts:
        with console.status(f"Testing: {prompt[:50]}...", spinner="dots"):
            result = client.generate(prompt, timeout=120)

        tokens = result.get("tokens", 0)
        time_s = result.get("time_s", 0)
        tps = result.get("tokens_per_sec", 0)
        total_tokens += tokens
        total_time += time_s
        console.print(f"  {tokens} tokens in {time_s:.1f}s ({tps:.1f} tok/s)")

    avg_tps = total_tokens / max(0.01, total_time)
    console.print(f"\n[bold]Average: {avg_tps:.1f} tokens/sec[/bold]")

    if avg_tps < 5:
        console.print("[yellow]Consider using a smaller model for better responsiveness.[/yellow]")
    elif avg_tps > 20:
        console.print("[green]Good performance! You could try a larger model.[/green]")


# ─── API Server ──────────────────────────────────────────────────────────────


@main.command()
@click.option("--port", "-p", default=8000, help="Port to listen on")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
def api(port: int, host: str):
    """Start an OpenAI-compatible API server.

    \b
    Other tools can connect at http://localhost:8000/v1/chat/completions
    """
    from forge.api.openai_compat import run_api_server

    console.print(
        Panel(
            f"Starting OpenAI-compatible API server\n\n"
            f"Base URL: [cyan]http://{host}:{port}/v1[/cyan]\n"
            f"Models:   [cyan]http://{host}:{port}/v1/models[/cyan]\n"
            f"Health:   [cyan]http://{host}:{port}/health[/cyan]\n\n"
            f"Use in your tools:\n"
            f'  OPENAI_BASE_URL=http://{host}:{port}/v1',
            title="ollama-forge API",
            border_style="blue",
        )
    )
    run_api_server(port=port, host=host)


# ─── Rules ───────────────────────────────────────────────────────────────────


@main.command("init-rules")
@click.option("--dir", "-d", "directory", default=".", help="Directory to create rules in")
def init_rules(directory: str):
    """Create a .forge-rules template in the current directory."""
    from forge.agents.rules import create_rules_template

    result = create_rules_template(directory)
    console.print(f"[green]{result}[/green]")


# ─── Codebase Indexing ──────────────────────────────────────────────────────


@main.command()
@click.option("--dir", "-d", "directory", default=".", help="Project directory to index")
@click.option("--summaries", is_flag=True, help="Generate LLM summaries for each file (slow)")
@click.option("--update", is_flag=True, help="Incremental update (only changed files)")
def index(directory: str, summaries: bool, update: bool):
    """Index the codebase for fast search and agent context retrieval.

    \b
    Creates a .forge/index/ directory with symbol and file metadata.
    Agents use this to find relevant code without manual file specification.
    """
    from forge.tools.codebase import CodebaseIndexer

    client = None
    if summaries:
        from forge.config import load_config
        from forge.hardware import detect_hardware, select_profile
        from forge.llm.client import OllamaClient

        config = load_config()
        hw = detect_hardware()
        profile = select_profile(hw)
        client = OllamaClient(
            model=config.default_model or profile.recommended_model,
            num_ctx=profile.num_ctx,
        )

    indexer = CodebaseIndexer(project_dir=directory, client=client)

    if update:
        with console.status("Updating index...", spinner="dots"):
            stats = indexer.update_index()
        console.print(
            f"[green]Index updated:[/green] "
            f"{stats['added']} added, {stats['updated']} updated, {stats['removed']} removed"
        )
    else:
        with console.status("Building index...", spinner="dots"):
            stats = indexer.build_index(generate_summaries=summaries)
        console.print(
            f"[green]Index built:[/green] {stats['files']} files, "
            f"{stats['symbols']} symbols in {stats['duration_s']}s"
        )

    # Show overview
    console.print(f"\n{indexer.get_project_overview(max_files=20)}")


@main.command()
@click.argument("query", nargs=-1)
@click.option("--dir", "-d", "directory", default=".", help="Project directory")
@click.option("--max-results", "-n", default=10, help="Maximum results")
def search(query: tuple, directory: str, max_results: int):
    """Search the codebase index for symbols, files, or content."""
    from forge.tools.codebase import CodebaseIndexer

    query_str = " ".join(query)
    if not query_str:
        console.print("Usage: forge search <query>")
        return

    indexer = CodebaseIndexer(project_dir=directory)
    results = indexer.search(query_str, max_results=max_results)

    if not results:
        console.print(f"No results for '{query_str}'. Run 'forge index' first if you haven't.")
        return

    table = Table(title=f"Search: {query_str}")
    table.add_column("Score", style="dim", width=5)
    table.add_column("File", style="cyan")
    table.add_column("Line", style="dim", width=5)
    table.add_column("Match")

    for r in results:
        table.add_row(
            f"{r.score:.1f}",
            r.file,
            str(r.line),
            r.content[:80],
        )
    console.print(table)


# ─── Undo ───────────────────────────────────────────────────────────────────


@main.command()
@click.option("--dir", "-d", "directory", default=".", help="Working directory")
def undo(directory: str):
    """Undo the last agent-made commit (safe revert, creates a new commit).

    \b
    Only reverts commits tagged with [forge]. Human commits are never touched.
    """
    from forge.tools.git import GitTool

    git = GitTool(working_dir=directory)

    # Show recent agent commits
    agent_commits = git.get_agent_commits(5)
    if not agent_commits:
        console.print("No agent commits found to undo.")
        return

    console.print("[bold]Recent agent commits:[/bold]")
    for c in agent_commits:
        console.print(f"  {c}")

    # Confirm
    if click.confirm("\nRevert the most recent agent commit?"):
        result = git._undo()
        console.print(f"[green]{result}[/green]")
    else:
        console.print("Cancelled.")


if __name__ == "__main__":
    main()
