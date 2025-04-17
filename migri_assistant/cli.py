import logging
from typing import List, Optional
from urllib.parse import urlparse

import typer

from migri_assistant.crawler.runner import ScrapyRunner
from migri_assistant.parsers.migri_parser import MigriParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress unnecessary warnings
logging.getLogger("onnxruntime").setLevel(logging.ERROR)  # Suppress ONNX warnings
logging.getLogger("transformers").setLevel(
    logging.ERROR
)  # Suppress potential transformers warnings
logging.getLogger("chromadb").setLevel(logging.WARNING)  # Reduce ChromaDB debug noise

app = typer.Typer(help="Migri Assistant CLI - Web crawling and parsing tool")


@app.command()
def crawl(
    url: str = typer.Argument(..., help="The URL to crawl content from"),
    depth: int = typer.Option(
        1,
        "--depth",
        "-d",
        help="Maximum link-following depth (1 is just the provided URL)",
    ),
    allowed_domains: Optional[List[str]] = typer.Option(
        None,
        "--domain",
        "-D",
        help="Domains to restrict crawling to (defaults to URL's domain)",
    ),
    output_dir: str = typer.Option(
        "crawled_content",
        "--output-dir",
        "-o",
        help="Directory to save crawled HTML files",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """
    Crawl a website to a configurable depth and save raw HTML content.

    The crawler is interruptible - press Ctrl+C to stop and save current progress.

    Example:
        $ python -m migri_assistant.cli crawl https://migri.fi -d 2 -o migri_content
    """
    # Set log level based on verbose flag
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Extract domain from URL if allowed_domains is not provided
    if allowed_domains is None:
        parsed_url = urlparse(url)
        allowed_domains = [parsed_url.netloc]

    typer.echo(f"🕸️ Starting web crawler on {url} with depth {depth}")
    typer.echo(f"💾 Saving HTML content to: {output_dir}")

    try:
        # Initialize crawler runner
        runner = ScrapyRunner()

        typer.echo("⚠️ Press Ctrl+C at any time to interrupt crawling.")

        # Start crawling
        results = runner.run(
            start_urls=[url],
            depth=depth,
            allowed_domains=allowed_domains,
            output_dir=output_dir,
        )

        # Output information
        typer.echo(f"✅ Crawling completed! Processed {len(results)} pages.")
        typer.echo(f"💾 Content saved as HTML files in {output_dir}")

    except KeyboardInterrupt:
        typer.echo("\n🛑 Crawling interrupted by user")
        typer.echo("✅ Partial results have been saved")
        typer.echo(f"💾 Crawled content saved to {output_dir}")
    except Exception as e:
        typer.echo(f"❌ Error during crawling: {str(e)}", err=True)
        raise typer.Exit(code=1)


@app.command()
def parse(
    input_dir: str = typer.Option(
        "crawled_content",
        "--input-dir",
        "-i",
        help="Directory containing HTML files to parse",
    ),
    output_dir: str = typer.Option(
        "parsed_content",
        "--output-dir",
        "-o",
        help="Directory to save parsed content as Markdown files",
    ),
    domain: Optional[str] = typer.Option(
        None,
        "--domain",
        "-d",
        help="Domain to parse (e.g. 'migri.fi'). If not provided, all domains are parsed.",
    ),
    site_type: str = typer.Option(
        "migri",
        "--site-type",
        "-s",
        help="Type of site to parse (determines which parser to use)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """
    Parse HTML files previously crawled and convert to structured Markdown.

    This command reads HTML files from the specified input directory,
    extracts meaningful content, and saves it as Markdown files.

    Example:
        $ python -m migri_assistant.cli parse -i crawled_content -o parsed_content -s migri
    """
    # Set log level based on verbose flag
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    typer.echo(f"📝 Starting HTML parsing from {input_dir}")
    typer.echo(f"📄 Saving parsed content to: {output_dir}")

    try:
        # Initialize appropriate parser based on site_type
        if site_type == "migri":
            typer.echo("🔧 Using specialized Migri.fi parser")
            parser = MigriParser(input_dir=input_dir, output_dir=output_dir)
        else:
            typer.echo(f"❌ Unsupported site type: {site_type}")
            raise typer.Exit(code=1)

        # Start parsing
        results = parser.parse_all(domain=domain)

        # Output information
        typer.echo(f"✅ Parsing completed! Processed {len(results)} files.")
        typer.echo(f"📝 Content saved as Markdown files in {output_dir}")
        typer.echo(f"📝 Index created at {output_dir}/{site_type}/index.md")

    except Exception as e:
        typer.echo(f"❌ Error during parsing: {str(e)}", err=True)
        raise typer.Exit(code=1)


@app.command()
def info():
    """Show information about the Migri Assistant and available commands."""
    typer.echo("Migri Assistant - Web crawling and parsing tool")
    typer.echo("\nAvailable commands:")
    typer.echo("  crawl     - Crawl websites and save HTML content")
    typer.echo("  parse     - Parse HTML files and convert to structured Markdown")
    typer.echo("  info      - Show this information")
    typer.echo("\nRun a command with --help for more information")


if __name__ == "__main__":
    app()
