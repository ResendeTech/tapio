import logging
import os

import typer

from tapio.config import ConfigManager
from tapio.config.settings import DEFAULT_CHROMA_COLLECTION, DEFAULT_CONTENT_DIR, DEFAULT_DIRS
from tapio.crawler.runner import CrawlerRunner
from tapio.parser import Parser
from tapio.vectorstore.vectorizer import MarkdownVectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress unnecessary warnings
logging.getLogger("onnxruntime").setLevel(logging.ERROR)  # Suppress ONNX warnings
logging.getLogger("transformers").setLevel(
    logging.ERROR,
)  # Suppress potential transformers warnings
logging.getLogger("chromadb").setLevel(logging.WARNING)  # Reduce ChromaDB debug noise


def find_sites_with_crawled_content(content_dir: str, crawled_subdir: str) -> list[str]:
    """Find all sites that have crawled HTML content.

    :param content_dir: The root content directory to search in
    :param crawled_subdir: The subdirectory name containing crawled files
    :return: List of site names that have crawled HTML content
    """
    crawled_sites: list[str] = []
    if not os.path.exists(content_dir):
        return crawled_sites

    for item in os.listdir(content_dir):
        item_path = os.path.join(content_dir, item)
        if os.path.isdir(item_path):
            crawled_path = os.path.join(item_path, crawled_subdir)
            if os.path.exists(crawled_path) and os.path.isdir(crawled_path):
                # Check if the crawled directory contains any HTML files
                has_html = any(f.endswith(".html") for root, _, files in os.walk(crawled_path) for f in files)
                if has_html:
                    crawled_sites.append(item)

    return crawled_sites


app = typer.Typer(help="Tapio Assistant CLI - Web crawling and parsing tool")


@app.command()
def crawl(
    site: str = typer.Argument(..., help="Site configuration to use for crawling (e.g., 'migri')"),
    depth: int | None = typer.Option(
        None,
        "--depth",
        "-d",
        help="Maximum link-following depth (if not specified, uses config file default)",
    ),
    config_path: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to custom parser configurations file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """
    Crawl a website to a configurable depth and save raw HTML content.

    This command takes a site identifier and uses the corresponding configuration
    from the site_configs.yaml file to determine the base URL for crawling.

    The crawler is interruptible - press Ctrl+C to stop and save current progress.

    Example:
        $ python -m tapio.cli crawl migri -d 2
    """
    # Set log level based on verbose flag
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Use ConfigManager for site configuration management
    try:
        config_manager = ConfigManager(config_path)
        available_sites = config_manager.list_available_sites()

        if site not in available_sites:
            typer.echo(f"❌ Unsupported site: {site}")
            typer.echo(f"Available sites: {', '.join(available_sites)}")
            raise typer.Exit(code=1)

        # Get the site configuration
        site_config = config_manager.get_site_config(site)
    except ValueError as e:
        typer.echo(f"❌ Error loading site configuration: {str(e)}")
        raise typer.Exit(code=1)

    # Get the base URL from the site configuration
    url = site_config.base_url

    # Implement depth precedence logic:
    # 1. Use user-provided value if given
    # 2. Otherwise, use config value (which could be default or explicitly set)
    if depth is not None:
        # User explicitly provided a depth value
        site_config.crawler_config.max_depth = depth

    # Construct the actual output directory path
    crawled_dir = os.path.join(DEFAULT_CONTENT_DIR, site, DEFAULT_DIRS["CRAWLED_DIR"])

    typer.echo(f"🕸️ Starting web crawler for {site} ({url}) with depth {site_config.crawler_config.max_depth}")
    typer.echo(f"💾 Saving HTML content to: {crawled_dir}")
    typer.echo(
        f"⏱️ Using {site_config.crawler_config.delay_between_requests}s delay between requests "
        f"and max {site_config.crawler_config.max_concurrent} concurrent requests",
    )

    try:
        # Initialize crawler runner
        runner = CrawlerRunner()

        typer.echo("⚠️ Press Ctrl+C at any time to interrupt crawling.")

        # Start crawling with simplified interface
        results = runner.run(site, site_config)

        # Output information
        typer.echo(f"✅ Crawling completed! Processed {len(results)} pages.")
        typer.echo(f"💾 Content saved as HTML files in {crawled_dir}")

    except KeyboardInterrupt:
        typer.echo("\n🛑 Crawling interrupted by user")
        typer.echo("✅ Partial results have been saved")
        typer.echo(f"💾 Crawled content saved to {crawled_dir}")
    except Exception as e:
        typer.echo(f"❌ Error during crawling: {str(e)}", err=True)
        raise typer.Exit(code=1)


@app.command()
def parse(
    site: str | None = typer.Argument(
        None,
        help="Site to parse (e.g., 'migri'). If not provided, all available sites with crawled content are parsed.",
    ),
    config_path: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to custom parser configurations file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """
    Parse HTML files previously crawled and convert to structured Markdown.

    This command reads HTML files from the specified input directory, extracts meaningful content
    based on site-specific configurations, and saves it as Markdown files with YAML frontmatter.

    Configurations define which XPath selectors to use for extracting content and how to convert
    HTML to Markdown for different website types.

    Examples:
        $ python -m tapio.cli parse migri
        $ python -m tapio.cli parse te_palvelut
        $ python -m tapio.cli parse kela --config custom_configs.yaml
    """
    # Set log level based on verbose flag
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    typer.echo(f"📝 Starting HTML parsing from {DEFAULT_DIRS['CRAWLED_DIR']}")
    typer.echo(f"📄 Saving parsed content to: {DEFAULT_DIRS['PARSED_DIR']}")

    try:
        # Use ConfigManager for site configuration management
        config_manager = ConfigManager(config_path)
        available_sites = config_manager.list_available_sites()

        if site is not None:
            # Parse a specific site
            if site in available_sites:
                typer.echo(f"🔧 Using configuration for site: {site}")
                parser = Parser(
                    site_name=site,
                    config_path=config_path,
                )
                results = parser.parse_all()

                # Output information
                typer.echo(f"✅ Parsing completed! Processed {len(results)} files.")
                parsed_dir = os.path.join(DEFAULT_CONTENT_DIR, site, DEFAULT_DIRS["PARSED_DIR"])
                typer.echo(f"📝 Content saved as Markdown files in {parsed_dir}")
                typer.echo(f"📝 Index created at {parsed_dir}/index.md")
            else:
                typer.echo(f"❌ Unsupported site: {site}")
                typer.echo(f"Available sites: {', '.join(available_sites)}")
                raise typer.Exit(code=1)
        else:
            # Parse all sites that have crawled content
            typer.echo("🔧 No site specified, parsing all available sites with crawled content")

            # Find which sites have crawled content by checking the content directory structure
            content_dir = DEFAULT_CONTENT_DIR
            if not os.path.exists(content_dir):
                typer.echo(f"❌ Content directory not found: {content_dir}")
                raise typer.Exit(code=1)

            # Get site directories that contain crawled content
            crawled_sites = find_sites_with_crawled_content(content_dir, DEFAULT_DIRS["CRAWLED_DIR"])

            if not crawled_sites:
                typer.echo("❌ No crawled content found to parse")
                raise typer.Exit(code=1)

            typer.echo(f"📂 Found crawled content for sites: {', '.join(crawled_sites)}")

            # Match crawled sites to available site configurations
            sites_to_parse: list[str] = []
            for site_name in available_sites:
                if site_name in crawled_sites:
                    sites_to_parse.append(site_name)

            if not sites_to_parse:
                typer.echo("❌ No site configurations found matching crawled content")
                typer.echo(f"Available sites: {', '.join(available_sites)}")
                typer.echo(f"Crawled sites: {', '.join(crawled_sites)}")
                raise typer.Exit(code=1)

            typer.echo(f"🎯 Parsing sites: {', '.join(sites_to_parse)}")

            # Parse each site
            total_results = []
            for site_name in sites_to_parse:
                typer.echo(f"🔧 Parsing site: {site_name}")
                parser = Parser(
                    site_name=site_name,
                    config_path=config_path,
                )

                site_results = parser.parse_all()
                total_results.extend(site_results)
                typer.echo(f"  ✅ {site_name}: Processed {len(site_results)} files")

            # Output summary information
            typer.echo(f"✅ All parsing completed! Processed {len(total_results)} files total.")
            typer.echo(f"📝 Content saved as Markdown files in {DEFAULT_CONTENT_DIR}")
            typer.echo(f"📊 Parsed {len(sites_to_parse)} sites: {', '.join(sites_to_parse)}")

    except Exception as e:
        typer.echo(f"❌ Error during parsing: {str(e)}", err=True)
        raise typer.Exit(code=1)


@app.command()
def vectorize(
    site: str | None = typer.Argument(
        None,
        help="Site to vectorize (e.g. 'migri'). If not provided, all sites are processed.",
    ),
    embedding_model: str = typer.Option(
        "all-MiniLM-L6-v2",
        "--model",
        "-m",
        help="Name of the sentence-transformers model to use",
    ),
    batch_size: int = typer.Option(
        20,
        "--batch-size",
        "-b",
        help="Number of documents to process in each batch",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """
    Vectorize parsed Markdown files and store in a vector database (ChromaDB).

    This command reads parsed Markdown files with frontmatter, generates embeddings,
    and stores them in ChromaDB with associated metadata from the original source.

    Examples:
        $ python -m tapio.cli vectorize migri
        $ python -m tapio.cli vectorize
    """
    # Set log level based on verbose flag
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    db_dir = DEFAULT_DIRS["CHROMA_DIR"]
    collection_name = DEFAULT_CHROMA_COLLECTION

    # Determine input directory based on site parameter
    if site is not None:
        # Process a specific site
        input_dir = os.path.join(DEFAULT_CONTENT_DIR, site, DEFAULT_DIRS["PARSED_DIR"])
        if not os.path.exists(input_dir):
            typer.echo(f"❌ No parsed content found for site: {site}")
            typer.echo(f"Expected directory: {input_dir}")
            raise typer.Exit(code=1)
        typer.echo(f"🧠 Starting vectorization of parsed content for site '{site}' from {input_dir}")
    else:
        # Process all sites
        input_dir = DEFAULT_CONTENT_DIR
        typer.echo(f"🧠 Starting vectorization of parsed content from all sites in {input_dir}")

    typer.echo(f"💾 Vector database will be stored in: {db_dir}")
    typer.echo(f"🔤 Using embedding model: {embedding_model}")
    typer.echo(f"📑 Using collection name: {collection_name}")

    try:
        # Initialize vectorizer
        vectorizer = MarkdownVectorizer(
            collection_name=collection_name,
            persist_directory=db_dir,
            embedding_model_name=embedding_model,
            chunk_size=1000,
            chunk_overlap=200,
        )

        # Process files in the directory
        typer.echo("⚙️ Processing markdown files...")
        # Process files without site filter (already handled by input_dir)
        count = vectorizer.process_directory(
            input_dir=input_dir,
            site_filter=None,
            batch_size=batch_size,
        )

        # Output information
        typer.echo(f"✅ Vectorization completed! Processed {count} files.")
        typer.echo(f"🔍 Vector database is ready for similarity search in {db_dir}")

    except Exception as e:
        typer.echo(f"❌ Error during vectorization: {str(e)}", err=True)
        raise typer.Exit(code=1)


@app.command()
def info(
    list_site_configs: bool = typer.Option(
        False,
        "--list-site-configs",
        "-l",
        help="List all available site configurations for parsing",
    ),
    show_site_config: str = typer.Option(
        None,
        "--show-site-config",
        "-s",
        help="Show detailed configuration for a specific site",
    ),
) -> None:
    """Show information about the Tapio Assistant and available commands."""
    # Use ConfigManager directly instead of going through Parser
    config_manager = ConfigManager()

    if list_site_configs:
        # List all available site configurations
        site_configs = config_manager.list_available_sites()
        typer.echo("Available site configurations for parsing:")
        for site_name in site_configs:
            typer.echo(f"  - {site_name}")
        return

    if show_site_config:
        # Show details for a specific site configuration
        try:
            config = config_manager.get_site_config(show_site_config)
            typer.echo(f"Configuration for site: {show_site_config}")
            typer.echo(f"  Base URL: {config.base_url}")
            typer.echo(f"  Base directory: {config.base_dir}")
            typer.echo(f"  Description: {config.description}")
            typer.echo("  Content selectors:")
            for selector in config.parser_config.content_selectors:
                typer.echo(f"    - {selector}")
            typer.echo(f"  Fallback to body: {config.parser_config.fallback_to_body}")
        except ValueError:
            typer.echo(f"Error: Site configuration '{show_site_config}' not found")
        return

    # Show general information
    typer.echo("Tapio Assistant - Web crawling and parsing tool")
    typer.echo("\nAvailable commands:")
    typer.echo("  crawl      - Crawl websites and save HTML content")
    typer.echo("  parse      - Parse HTML files and convert to structured Markdown")
    typer.echo("  vectorize  - Vectorize parsed Markdown files and store in ChromaDB")
    typer.echo("  tapio-app  - Launch the Tapio web interface for querying with the RAG chatbot")
    typer.echo("  info       - Show this information")
    typer.echo("  dev        - Launch the development server for the Tapio Assistant chatbot")
    typer.echo("\nRun a command with --help for more information")


@app.command()
def tapio_app(
    model_name: str = typer.Option(
        "llama3.2:latest",
        "--model-name",
        "-m",
        help="Ollama model to use for LLM inference",
    ),
    max_tokens: int = typer.Option(
        1024,
        "--max-tokens",
        "-t",
        help="Maximum number of tokens to generate",
    ),
    share: bool = typer.Option(
        False,
        "--share",
        help="Create a shareable link for the app",
    ),
) -> None:
    """Launch the Tapio web interface for RAG-powered chatbot."""
    try:
        # Import the main function from the gradio_app module
        from tapio.app import main as launch_app

        collection_name = DEFAULT_CHROMA_COLLECTION
        db_dir = DEFAULT_DIRS["CHROMA_DIR"]

        typer.echo(f"🚀 Starting Gradio app with {model_name} model")
        typer.echo(f"📚 Using ChromaDB collection '{collection_name}' from '{db_dir}'")

        if share:
            typer.echo("🔗 Creating a shareable link")

        # Launch the Gradio app with CLI parameters
        launch_app(
            collection_name=collection_name,
            persist_directory=db_dir,
            model_name=model_name,
            max_tokens=max_tokens,
            share=share,
        )

    except ImportError as e:
        typer.echo(f"❌ Error importing Gradio: {str(e)}", err=True)
        typer.echo("Make sure Gradio is installed with 'uv add gradio'")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"❌ Error launching Gradio app: {str(e)}", err=True)
        raise typer.Exit(code=1)


@app.command()
def adk_server(
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port to bind the ADK server to",
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help="Host to bind the ADK server to",
    ),
    model_name: str = typer.Option(
        None,
        "--model-name",
        "-m",
        help="Default LLM model to use for the agent (auto-detects if not specified)",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable auto-reload for development",
    ),
    disable_web_ui: bool = typer.Option(
        False,
        "--disable-web-ui",
        help="Disable the built-in web development UI",
    ),
) -> None:
    """Launch the ADK-based Tapio Assistant server."""
    try:
        from tapio.adk_server import launch_tapio_adk_server
        import os
        from pathlib import Path

        # Get the agents directory
        current_dir = Path(__file__).parent
        agents_dir = str(current_dir / "agents")

        # Check if we have required data
        db_dir = DEFAULT_DIRS["CHROMA_DIR"]
        if not os.path.exists(db_dir):
            typer.echo("❌ No vector database found. Please run the vectorization pipeline first:", err=True)
            typer.echo("   1. uv run tapio crawl <site>")
            typer.echo("   2. uv run tapio parse <site>")
            typer.echo("   3. uv run tapio vectorize")
            raise typer.Exit(code=1)

        # Check model availability based on provider
        from tapio.config.model_config import get_model_config
        from tapio.config.settings import has_gemini_api_key, get_default_model
        
        # Auto-detect model if not specified
        if model_name is None:
            model_name = get_default_model()
            typer.echo(f"🤖 Auto-detected model: {model_name}")
        
        model_config = get_model_config(model_name)
        
        if model_config.provider == "gemini":
            if not has_gemini_api_key():
                typer.echo("❌ Gemini API key required for Gemini models.", err=True)
                typer.echo("Please set one of the following environment variables:")
                typer.echo("  - GEMINI_API_KEY")
                typer.echo("  - GOOGLE_API_KEY") 
                typer.echo("  - GOOGLE_AI_API_KEY")
                typer.echo("\nGet your API key from: https://aistudio.google.com/app/apikey")
                raise typer.Exit(code=1)
            typer.echo(f"✅ Using Gemini model '{model_name}' with API key")
            
        elif model_config.provider == "ollama":
            typer.echo(f"🔍 Checking if Ollama model '{model_name}' is available...")
            try:
                import ollama
                ollama.show(model_name)
                typer.echo(f"✅ Model '{model_name}' is available")
            except Exception:
                typer.echo(f"⚠️  Model '{model_name}' not found. Installing...")
                try:
                    ollama.pull(model_name)
                    typer.echo(f"✅ Model '{model_name}' installed successfully")
                except Exception as e:
                    typer.echo(f"❌ Failed to install model: {e}", err=True)
                    typer.echo(f"Please run: ollama pull {model_name}")
                    raise typer.Exit(code=1)

        typer.echo(f"🚀 Starting Tapio ADK Server...")
        typer.echo(f"📁 Agents directory: {agents_dir}")
        typer.echo(f"🤖 Default model: {model_name} (provider: {model_config.provider})")
        typer.echo(f"🌐 Server will be available at: http://{host}:{port}")
        
        if not disable_web_ui:
            typer.echo(f"🌍 Web development UI: http://{host}:{port}")
        
        typer.echo(f"📖 API documentation: http://{host}:{port}/docs")

        # Set environment variable for model name
        os.environ["TAPIO_DEFAULT_MODEL"] = model_name

        # Launch the server
        launch_tapio_adk_server(
            agents_dir=agents_dir,
            host=host,
            port=port,
            enable_web_ui=not disable_web_ui,
            reload=reload,
        )

    except ImportError as e:
        typer.echo(f"❌ Error importing ADK components: {str(e)}", err=True)
        typer.echo("Make sure google-adk is installed: uv add google-adk")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"❌ Error launching ADK server: {str(e)}", err=True)
        raise typer.Exit(code=1)


@app.command()
def list_models() -> None:
    """List available LLM models for the ADK agent."""
    try:
        from tapio.config.model_config import list_available_models
        
        models = list_available_models()
        
        typer.echo("🤖 Available LLM models:")
        typer.echo("")
        
        for model_name, config in models.items():
            provider_emoji = "🏠" if config.provider == "ollama" else "☁️"
            typer.echo(f"{provider_emoji} {model_name}")
            typer.echo(f"   Provider: {config.provider}")
            typer.echo(f"   Max tokens: {config.max_tokens}")
            typer.echo(f"   Context window: {config.context_window}")
            typer.echo("")
            
        typer.echo("💡 Use any model name with: uv run -m tapio.cli adk-server --model-name <model>")
        
    except Exception as e:
        typer.echo(f"❌ Error listing models: {str(e)}", err=True)
        raise typer.Exit(code=1)


@app.command()
def dev() -> None:
    """Launch the development server for the Tapio Assistant chatbot."""
    typer.echo("🚀 Launching Tapio Assistant chatbot development server...")
    # Call the adk_server function with development settings
    adk_server(
        model_name="llama3.2",
        host="127.0.0.1",
        port=8000,
        reload=True,
        web_ui=True,
    )


@app.command()
def list_sites(
    config_path: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to custom parser configurations file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed information about each site configuration",
    ),
) -> None:
    """
    List available site configurations for the parser.

    This command lists all the available sites that can be used with the parse command.
    Use the --verbose flag to see detailed information about each site's configuration.
    """
    try:
        # Use ConfigManager directly for better configuration handling
        config_manager = ConfigManager(config_path)
        available_sites = config_manager.list_available_sites()

        typer.echo("📋 Available Site Configurations:")

        for site_name in available_sites:
            if verbose:
                try:
                    # Get detailed configuration for the site
                    site_config = config_manager.get_site_config(site_name)
                    typer.echo(f"\n📄 {site_name}:")
                    typer.echo(f"  Description: {site_config.description or 'No description'}")
                    typer.echo(f"  Title selector: {site_config.parser_config.title_selector}")
                    typer.echo("  Content selectors:")
                    for selector in site_config.parser_config.content_selectors:
                        typer.echo(f"    - {selector}")
                    typer.echo(f"  Fallback to body: {site_config.parser_config.fallback_to_body}")
                    typer.echo("  Crawler configuration:")
                    typer.echo(f"    - Delay between requests: {site_config.crawler_config.delay_between_requests}s")
                    typer.echo(f"    - Max concurrent requests: {site_config.crawler_config.max_concurrent}")
                except ValueError:
                    # Skip sites with invalid configurations
                    typer.echo(f"\n❌ {site_name}: Invalid configuration")
            else:
                # Simpler output for non-verbose mode
                site_descriptions = config_manager.get_site_descriptions()
                description = f" - {site_descriptions[site_name]}" if site_name in site_descriptions else ""
                typer.echo(f"  • {site_name}{description}")

        typer.echo("\nUse these sites with the parse command, e.g.:")
        typer.echo(f"  $ python -m tapio.cli parse {available_sites[0]}")

    except Exception as e:
        typer.echo(f"❌ Error listing site configurations: {str(e)}", err=True)
        raise typer.Exit(code=1)


@app.command()
def setup_api_keys() -> None:
    """Interactive setup for API keys."""
    typer.echo("🔧 Tapio API Key Setup")
    typer.echo("=" * 30)
    
    from tapio.config.settings import has_gemini_api_key, get_gemini_api_key
    
    # Check current Gemini API key status
    if has_gemini_api_key():
        current_key = get_gemini_api_key()
        masked_key = f"{current_key[:8]}...{current_key[-4:]}" if current_key else "None"
        typer.echo(f"✅ Gemini API key already set: {masked_key}")
        
        if not typer.confirm("Do you want to update it?"):
            typer.echo("Keeping existing API key.")
        else:
            _setup_gemini_key()
    else:
        typer.echo("❌ No Gemini API key found.")
        if typer.confirm("Do you want to set up a Gemini API key?"):
            _setup_gemini_key()
        else:
            typer.echo("Skipping Gemini API key setup.")
    
    typer.echo("\n🎉 Setup complete!")
    typer.echo("You can now use Gemini models with:")
    typer.echo("  uv run tapio adk-server --model-name gemini-2.0-flash")


def _setup_gemini_key() -> None:
    """Setup Gemini API key."""
    typer.echo("\n🔑 Setting up Gemini API Key")
    typer.echo("Get your API key from: https://aistudio.google.com/app/apikey")
    
    api_key = typer.prompt("Enter your Gemini API key", hide_input=True)
    
    if not api_key or len(api_key.strip()) < 10:
        typer.echo("❌ Invalid API key. Please try again.", err=True)
        return
    
    # Test the API key
    typer.echo("🔍 Testing API key...")
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key.strip())
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Hello! Just testing the API key.")
        typer.echo("✅ API key is valid!")
        
        # Save to environment (for this session)
        import os
        os.environ["GEMINI_API_KEY"] = api_key.strip()
        
        # Provide instructions for permanent setup
        typer.echo("\n💡 To make this permanent, add to your shell profile:")
        typer.echo(f"export GEMINI_API_KEY='{api_key.strip()}'")
        typer.echo("\nOr create a .env file in your project root:")
        typer.echo(f"GEMINI_API_KEY={api_key.strip()}")
        
    except ImportError:
        typer.echo("❌ google-generativeai package not installed.", err=True)
        typer.echo("Run: uv add google-generativeai")
    except Exception as e:
        typer.echo(f"❌ API key test failed: {e}", err=True)
        typer.echo("Please check your API key and try again.")


def run_tapio_app() -> None:
    """Entry point for the 'dev' command to launch the Tapio app with default settings."""
    # This function calls the tapio_app command with default settings
    tapio_app(
        model_name="llama3.2",
        share=False,
    )


if __name__ == "__main__":
    app()
