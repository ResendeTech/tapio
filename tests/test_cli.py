"""Tests for the CLI module."""

import os
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from tapio.cli import app
from tapio.config.settings import DEFAULT_CHROMA_COLLECTION, DEFAULT_CONTENT_DIR, DEFAULT_DIRS


@pytest.fixture
def runner():
    """Fixture for creating a CLI runner."""
    return CliRunner()


class TestCli:
    """Tests for the CLI module."""

    def test_info_command(self, runner):
        """Test the info command."""
        result = runner.invoke(app, ["info"])

        # Check that the command ran successfully
        assert result.exit_code == 0

        # Check that expected output is present
        assert "Tapio Assistant" in result.stdout
        assert "Available commands:" in result.stdout
        assert "crawl" in result.stdout
        assert "parse" in result.stdout
        assert "vectorize" in result.stdout
        assert "info" in result.stdout

    @patch("tapio.cli.CrawlerRunner")
    @patch("tapio.cli.ConfigManager")
    def test_crawl_command(self, mock_config_manager, mock_crawler_runner, runner):
        """Test the crawl command."""
        # Set up mocks
        mock_runner_instance = MagicMock()
        mock_runner_instance.run.return_value = ["page1", "page2", "page3"]
        mock_crawler_runner.return_value = mock_runner_instance

        # Mock ConfigManager
        mock_config_instance = MagicMock()
        mock_site_config = MagicMock()
        mock_site_config.base_url = "https://example.com"
        mock_site_config.base_dir = "example.com"  # This should be just the domain
        # Mock the crawler_config with appropriate default values
        mock_crawler_config = MagicMock()
        mock_crawler_config.delay_between_requests = 1.0
        mock_crawler_config.max_concurrent = 5
        mock_crawler_config.depth = 1  # Add depth attribute
        mock_site_config.crawler_config = mock_crawler_config
        mock_config_instance.get_site_config.return_value = mock_site_config
        mock_config_instance.list_available_sites.return_value = ["migri"]
        mock_config_manager.return_value = mock_config_instance

        # Run the command
        result = runner.invoke(
            app,
            [
                "crawl",
                "migri",
                "--depth",
                "2",
            ],
        )

        # Check that the command ran successfully
        assert result.exit_code == 0

        # Check that list_available_sites was called
        mock_config_instance.list_available_sites.assert_called_once()

        # Check that get_site_config was called with the correct site name
        mock_config_instance.get_site_config.assert_called_once_with("migri")

        # Check that the runner was initialized correctly
        mock_crawler_runner.assert_called_once()

        # Check that run was called with the new interface
        mock_runner_instance.run.assert_called_once_with("migri", mock_site_config)

        # Check that depth was overridden
        assert mock_site_config.crawler_config.max_depth == 2

        # Check expected output in stdout
        assert "Starting web crawler" in result.stdout
        assert "Crawling completed" in result.stdout
        assert "Processed 3 pages" in result.stdout

    @patch("tapio.cli.CrawlerRunner")
    @patch("tapio.cli.ConfigManager")
    def test_crawl_command_keyboard_interrupt(self, mock_config_manager, mock_crawler_runner, runner):
        """Test handling of keyboard interrupt in crawl command."""
        # Mock ConfigManager
        mock_config_instance = MagicMock()
        mock_site_config = MagicMock()
        mock_site_config.base_url = "https://example.com"
        mock_site_config.base_dir = "example.com"
        # Mock the crawler_config
        mock_crawler_config = MagicMock()
        mock_crawler_config.delay_between_requests = 1.0
        mock_crawler_config.max_concurrent = 5
        mock_crawler_config.depth = 1
        mock_site_config.crawler_config = mock_crawler_config
        mock_config_instance.get_site_config.return_value = mock_site_config
        mock_config_instance.list_available_sites.return_value = ["migri"]
        mock_config_manager.return_value = mock_config_instance

        # Set up mock to raise KeyboardInterrupt
        mock_runner_instance = MagicMock()
        mock_runner_instance.run.side_effect = KeyboardInterrupt()
        mock_crawler_runner.return_value = mock_runner_instance

        # Run the command
        result = runner.invoke(app, ["crawl", "migri"])

        # Check that the command exited successfully (handled the interrupt)
        assert result.exit_code == 0

        # Check expected output in stdout
        assert "Starting web crawler for migri" in result.stdout
        assert "Crawling interrupted by user" in result.stdout
        assert "Partial results have been saved" in result.stdout

    @patch("tapio.cli.CrawlerRunner")
    @patch("tapio.cli.ConfigManager")
    def test_crawl_command_exception(self, mock_config_manager, mock_crawler_runner, runner):
        """Test handling of exceptions in crawl command."""
        # Mock ConfigManager
        mock_config_instance = MagicMock()
        mock_site_config = MagicMock()
        mock_site_config.base_url = "https://example.com"
        mock_site_config.base_dir = "example.com"
        # Mock the crawler_config
        mock_crawler_config = MagicMock()
        mock_crawler_config.delay_between_requests = 1.0
        mock_crawler_config.max_concurrent = 5
        mock_crawler_config.depth = 1
        mock_site_config.crawler_config = mock_crawler_config
        mock_config_instance.get_site_config.return_value = mock_site_config
        mock_config_instance.list_available_sites.return_value = ["migri"]
        mock_config_manager.return_value = mock_config_instance

        # Set up mock to raise an exception
        mock_runner_instance = MagicMock()
        mock_runner_instance.run.side_effect = Exception("Test error")
        mock_crawler_runner.return_value = mock_runner_instance

        # Run the command
        result = runner.invoke(app, ["crawl", "migri"])

        # Check that the command exited with error code
        assert result.exit_code == 1

        # Check expected output in stdout
        assert "Starting web crawler for migri" in result.stdout
        assert "Error during crawling: Test error" in result.stdout

    @patch("tapio.cli.ConfigManager")
    def test_crawl_command_invalid_site(self, mock_config_manager, runner):
        """Test the crawl command with an invalid site name."""
        # Mock the ConfigManager
        mock_config_instance = MagicMock()
        mock_config_instance.list_available_sites.return_value = ["migri", "te_palvelut", "kela"]
        mock_config_manager.return_value = mock_config_instance

        # Run the command with an unsupported site
        result = runner.invoke(app, ["crawl", "unsupported_site"])

        # Check that the command exited with error code
        assert result.exit_code == 1

        # Check expected output in stdout
        assert "Unsupported site: unsupported_site" in result.stdout
        assert "Available sites: migri, te_palvelut, kela" in result.stdout

    @patch("tapio.cli.ConfigManager")
    @patch("tapio.cli.Parser")
    def test_parse_command(self, mock_parser, mock_config_manager, runner):
        """Test the parse command."""
        # Set up mock parser
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_all.return_value = ["file1", "file2", "file3"]
        mock_parser.return_value = mock_parser_instance

        # Set up mock config manager
        mock_config_instance = MagicMock()
        mock_config_instance.list_available_sites.return_value = ["migri"]
        mock_config_manager.return_value = mock_config_instance

        # Run the command
        result = runner.invoke(
            app,
            [
                "parse",
                "--site",
                "migri",
            ],
        )

        # Check that the command ran successfully
        assert result.exit_code == 0

        # Check that the parser was initialized correctly
        mock_parser.assert_called_once_with(
            site_name="migri",
            config_path=None,
        )

        # Check that list_available_sites was called
        mock_config_instance.list_available_sites.assert_called_once()

        # Check that parse_all was called correctly (without domain parameter)
        mock_parser_instance.parse_all.assert_called_once_with()

        # Check expected output in stdout
        assert "Starting HTML parsing" in result.stdout
        assert "Using configuration for site: migri" in result.stdout
        assert "Parsing completed" in result.stdout
        assert "Processed 3 files" in result.stdout

    @patch("tapio.cli.ConfigManager")
    def test_parse_command_unsupported_site(self, mock_config_manager, runner):
        """Test the parse command with an unsupported site."""
        # Set up mock config manager
        mock_config_instance = MagicMock()
        mock_config_instance.list_available_sites.return_value = ["migri", "te_palvelut", "kela"]
        mock_config_manager.return_value = mock_config_instance

        # Run the command with an unsupported site
        result = runner.invoke(app, ["parse", "--site", "unsupported"])

        # Check that the command exited with error code
        assert result.exit_code == 1

        # Check expected output in stdout
        assert "Unsupported site: unsupported" in result.stdout
        assert "Available sites: migri, te_palvelut, kela" in result.stdout

        # Check that list_available_sites was called
        mock_config_instance.list_available_sites.assert_called_once()

    @patch("tapio.cli.ConfigManager")
    @patch("tapio.cli.Parser")
    def test_parse_command_exception(self, mock_parser, mock_config_manager, runner):
        """Test handling of exceptions in parse command."""
        # Set up mock parser that raises an exception
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_all.side_effect = Exception("Test error")
        mock_parser.return_value = mock_parser_instance

        # Set up mock config manager
        mock_config_instance = MagicMock()
        mock_config_instance.list_available_sites.return_value = ["migri"]
        mock_config_manager.return_value = mock_config_instance

        # Run the command
        result = runner.invoke(app, ["parse", "--site", "migri"])

        # Check that the command exited with error code
        assert result.exit_code == 1

        # Check expected output in stdout
        assert "Starting HTML parsing" in result.stdout
        assert "Error during parsing: Test error" in result.stdout

        # Check that list_available_sites was called
        mock_config_instance.list_available_sites.assert_called_once()

    @patch("tapio.cli.ConfigManager")
    @patch("tapio.cli.Parser")
    @patch("os.path.exists")
    def test_parse_command_custom_config(self, mock_exists, mock_parser, mock_config_manager, runner):
        """Test the parse command with a custom config path."""
        # Setup mock for file existence check
        mock_exists.return_value = True

        # Set up mock parser
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_all.return_value = ["file1", "file2"]
        mock_parser.return_value = mock_parser_instance

        # Set up mock config manager
        mock_config_instance = MagicMock()
        mock_config_instance.list_available_sites.return_value = ["custom_site"]
        mock_config_manager.return_value = mock_config_instance

        # Run the command with a custom config
        result = runner.invoke(
            app,
            [
                "parse",
                "--site",
                "custom_site",
                "--config",
                "custom_configs.yaml",
            ],
        )

        # Check that the command ran successfully
        assert result.exit_code == 0

        # Check that ConfigManager was instantiated with the correct custom config path
        mock_config_manager.assert_called_with("custom_configs.yaml")

        # Check that list_available_sites was called
        mock_config_instance.list_available_sites.assert_called_once()

        # Check that the parser was initialized correctly with the custom config
        mock_parser.assert_called_once_with(
            site_name="custom_site",
            config_path="custom_configs.yaml",
        )

        # Check that parse_all was called correctly (without domain parameter)
        mock_parser_instance.parse_all.assert_called_once_with()

    @patch("tapio.cli.MarkdownVectorizer")
    def test_vectorize_command(self, mock_vectorizer, runner):
        """Test the vectorize command."""
        # Set up mock
        mock_vectorizer_instance = MagicMock()
        mock_vectorizer_instance.process_directory.return_value = 5
        mock_vectorizer.return_value = mock_vectorizer_instance

        # Run the command
        result = runner.invoke(
            app,
            [
                "vectorize",
            ],
        )

        # Check that the command ran successfully
        assert result.exit_code == 0

        # Check that the vectorizer was initialized correctly
        mock_vectorizer.assert_called_once_with(
            collection_name=DEFAULT_CHROMA_COLLECTION,
            persist_directory=DEFAULT_DIRS["CHROMA_DIR"],
            embedding_model_name="all-MiniLM-L6-v2",
            chunk_size=1000,
            chunk_overlap=200,
        )

        # Check that process_directory was called correctly
        mock_vectorizer_instance.process_directory.assert_called_once_with(
            input_dir=DEFAULT_CONTENT_DIR,
            site_filter=None,
            batch_size=20,
        )

        # Check expected output in stdout
        assert "Starting vectorization" in result.stdout
        assert f"Vector database will be stored in: {DEFAULT_DIRS['CHROMA_DIR']}" in result.stdout
        assert "Using embedding model: all-MiniLM-L6-v2" in result.stdout
        assert "Vectorization completed" in result.stdout
        assert "Processed 5 files" in result.stdout

    @patch("tapio.cli.MarkdownVectorizer")
    def test_vectorize_command_with_site(self, mock_vectorizer, runner):
        """Test the vectorize command with site filter."""
        # Set up mock
        mock_vectorizer_instance = MagicMock()
        mock_vectorizer_instance.process_directory.return_value = 3
        mock_vectorizer.return_value = mock_vectorizer_instance

        # Mock os.path.exists to return True for the site directory
        with patch("tapio.cli.os.path.exists", return_value=True):
            # Run the command with site filter
            result = runner.invoke(app, ["vectorize", "--site", "migri"])

        # Check that the command ran successfully
        assert result.exit_code == 0

        # Check that process_directory was called with the site filter
        expected_input_dir = os.path.join(DEFAULT_CONTENT_DIR, "migri", DEFAULT_DIRS["PARSED_DIR"])
        mock_vectorizer_instance.process_directory.assert_called_once_with(
            input_dir=expected_input_dir,
            site_filter=None,
            batch_size=20,
        )

        # Check expected output in stdout
        assert "Starting vectorization" in result.stdout
        assert "Vectorization completed" in result.stdout
        assert "Processed 3 files" in result.stdout

    @patch("tapio.cli.MarkdownVectorizer")
    def test_vectorize_command_exception(self, mock_vectorizer, runner):
        """Test handling of exceptions in vectorize command."""
        # Set up mock to raise an exception
        mock_vectorizer_instance = MagicMock()
        mock_vectorizer_instance.process_directory.side_effect = Exception("Test error")
        mock_vectorizer.return_value = mock_vectorizer_instance

        # Run the command
        result = runner.invoke(app, ["vectorize"])

        # Check that the command exited with error code
        assert result.exit_code == 1

        # Check expected output in stdout
        assert "Starting vectorization" in result.stdout
        assert "Error during vectorization: Test error" in result.stdout

    @patch("tapio.app.main")
    def test_tapio_app_command(self, mock_launch_gradio, runner):
        """Test the tapio-app command."""
        # Run the command
        result = runner.invoke(app, ["tapio-app"])

        # Check that the command ran successfully
        assert result.exit_code == 0

        # Check that launch_gradio was called correctly
        mock_launch_gradio.assert_called_once_with(
            collection_name=DEFAULT_CHROMA_COLLECTION,
            persist_directory=DEFAULT_DIRS["CHROMA_DIR"],
            model_name="llama3.2:latest",
            max_tokens=1024,
            share=False,
        )

    @patch("tapio.app.main")
    def test_tapio_app_command_with_options(self, mock_launch_gradio, runner):
        """Test the tapio-app command with custom options."""
        # Run the command with options
        result = runner.invoke(
            app,
            [
                "tapio-app",
                "--model-name",
                "llama3.2:latest",
                "--max-tokens",
                "2048",
                "--share",
            ],
        )

        # Check that the command ran successfully
        assert result.exit_code == 0

        # Check that launch_gradio was called correctly with the custom options
        mock_launch_gradio.assert_called_once_with(
            collection_name=DEFAULT_CHROMA_COLLECTION,
            persist_directory=DEFAULT_DIRS["CHROMA_DIR"],
            model_name="llama3.2:latest",
            max_tokens=2048,
            share=True,
        )

    @patch("tapio.cli.tapio_app")
    def test_dev_command(self, mock_tapio_app, runner):
        """Test the dev command."""
        # Run the command
        result = runner.invoke(app, ["dev"])

        # Check that the command ran successfully
        assert result.exit_code == 0

        # Check that tapio_app was called correctly
        mock_tapio_app.assert_called_once_with(
            model_name="llama3.2",
            share=False,
        )

        # Check expected output in stdout
        assert "Launching Tapio Assistant chatbot development server" in result.stdout

    @patch("tapio.cli.ConfigManager")
    def test_list_sites_command(self, mock_config_manager, runner):
        """Test the list-sites command."""
        # Set up mock config manager
        mock_config_instance = MagicMock()
        mock_config_instance.list_available_sites.return_value = ["migri", "te_palvelut", "kela"]
        mock_config_instance.get_site_descriptions.return_value = {
            "migri": "Finnish Immigration Service",
            "te_palvelut": "Employment Services",
            "kela": "Social Insurance Institution",
        }
        mock_config_manager.return_value = mock_config_instance

        # Run the command
        result = runner.invoke(app, ["list-sites"])

        # Check that the command ran successfully
        assert result.exit_code == 0

        # Check that list_available_sites was called
        mock_config_instance.list_available_sites.assert_called_once()

        # Check that get_site_descriptions was called (may be called multiple times)
        assert mock_config_instance.get_site_descriptions.called

        # Check expected output in stdout
        assert "Available Site Configurations:" in result.stdout
        assert "migri" in result.stdout
        assert "te_palvelut" in result.stdout
        assert "kela" in result.stdout
        assert "Finnish Immigration Service" in result.stdout

    @patch("tapio.cli.ConfigManager")
    def test_list_sites_command_verbose(self, mock_config_manager, runner):
        """Test the list-sites command with verbose flag."""
        # Set up mock config manager
        mock_config_instance = MagicMock()
        mock_config_instance.list_available_sites.return_value = ["migri"]

        # Mock site config
        mock_site_config = MagicMock()
        mock_site_config.description = "Finnish Immigration Service"

        # Mock parser config within site config
        mock_parser_config = MagicMock()
        mock_parser_config.title_selector = "h1"
        mock_parser_config.content_selectors = ["main", "article"]
        mock_parser_config.fallback_to_body = True
        mock_site_config.parser_config = mock_parser_config

        mock_config_instance.get_site_config.return_value = mock_site_config
        mock_config_manager.return_value = mock_config_instance

        # Run the command with verbose flag
        result = runner.invoke(app, ["list-sites", "--verbose"])

        # Check that the command ran successfully
        assert result.exit_code == 0

        # Check that list_available_sites was called
        mock_config_instance.list_available_sites.assert_called_once()

        # Check that get_site_config was called with the right site
        mock_config_instance.get_site_config.assert_called_with("migri")

        # Check expected output in stdout
        assert "Available Site Configurations:" in result.stdout
        assert "Description: Finnish Immigration Service" in result.stdout
        assert "Title selector: h1" in result.stdout
        assert "Content selectors:" in result.stdout
        assert "Fallback to body: True" in result.stdout

    @patch("tapio.cli.ConfigManager")
    def test_list_sites_command_exception(self, mock_config_manager, runner):
        """Test handling of exceptions in list-sites command."""
        # Set up mock to raise an exception
        mock_config_instance = MagicMock()
        mock_config_instance.list_available_sites.side_effect = Exception("Test error")
        mock_config_manager.return_value = mock_config_instance

        # Run the command
        result = runner.invoke(app, ["list-sites"])

        # Check that the command exited with error code
        assert result.exit_code == 1

        # Check expected output in stdout
        assert "Error listing site configurations: Test error" in result.stdout

    @patch("tapio.cli.ConfigManager")
    @patch("tapio.cli.Parser")
    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("os.walk")
    def test_parse_command_no_site_specified(
        self,
        mock_walk,
        mock_isdir,
        mock_listdir,
        mock_exists,
        mock_parser,
        mock_config_manager,
        runner,
    ):
        """Test the parse command when no site is specified - should parse all available sites with crawled content."""
        # Setup mocks for directory structure - new structure uses content/site_name/crawled/
        mock_exists.side_effect = lambda path: (
            path == DEFAULT_CONTENT_DIR or "migri/crawled" in path or "kela/crawled" in path or "vero/crawled" in path
        )
        mock_listdir.return_value = ["migri", "kela", "vero", "parsed"]
        mock_isdir.side_effect = lambda path: not path.endswith(".json")

        # Mock os.walk to return HTML files for each site's crawled directory
        def mock_walk_side_effect(path):
            if "migri/crawled" in path:
                return [("/content/migri/crawled", [], ["page1.html", "page2.html"])]
            elif "kela/crawled" in path:
                return [("/content/kela/crawled", [], ["page3.html"])]
            elif "vero/crawled" in path:
                return [("/content/vero/crawled", [], ["page4.html", "page5.html"])]
            return []

        mock_walk.side_effect = mock_walk_side_effect

        # Set up mock parser instances
        mock_parser_instances = []
        for i in range(3):  # For 3 sites
            mock_instance = MagicMock()
            mock_instance.parse_all.return_value = [f"file{i * 2 + 1}", f"file{i * 2 + 2}"]
            mock_parser_instances.append(mock_instance)

        mock_parser.side_effect = mock_parser_instances

        # Set up mock config manager - sites match crawled site directories directly
        mock_config_instance = MagicMock()
        mock_config_instance.list_available_sites.return_value = ["migri", "kela", "vero"]
        mock_config_manager.return_value = mock_config_instance

        # Run the command without --site parameter
        result = runner.invoke(app, ["parse"])

        # Check that the command ran successfully
        assert result.exit_code == 0

        # Check that all three parsers were created
        assert mock_parser.call_count == 3

        # Check expected calls to Parser constructor - new format uses site_name only
        expected_calls = [
            ("migri", None),
            ("kela", None),
            ("vero", None),
        ]

        for i, (site_name, config_path) in enumerate(expected_calls):
            assert mock_parser.call_args_list[i][1]["site_name"] == site_name
            assert mock_parser.call_args_list[i][1]["config_path"] == config_path

        # Check that parse_all was called for each parser
        for mock_instance in mock_parser_instances:
            mock_instance.parse_all.assert_called_once_with()

        # Check expected output in stdout
        assert "No site specified, parsing all available sites with crawled content" in result.stdout
        assert "Found crawled content for sites: migri, kela, vero" in result.stdout
        assert "Parsing sites: migri, kela, vero" in result.stdout
        assert "Parsing site: migri" in result.stdout
        assert "Parsing site: kela" in result.stdout
        assert "Parsing site: vero" in result.stdout
        assert "All parsing completed! Processed 6 files total." in result.stdout
        assert "Parsed 3 sites: migri, kela, vero" in result.stdout

    @patch("tapio.cli.ConfigManager")
    @patch("os.path.exists")
    def test_parse_command_no_site_crawled_dir_not_found(self, mock_exists, mock_config_manager, runner):
        """Test the parse command when no site is specified and crawled directory doesn't exist."""
        # Setup mocks
        mock_exists.return_value = False

        mock_config_instance = MagicMock()
        mock_config_instance.list_available_sites.return_value = ["migri", "kela"]
        mock_config_manager.return_value = mock_config_instance

        # Run the command without --site parameter
        result = runner.invoke(app, ["parse"])

        # Check that the command exited with error code
        assert result.exit_code == 1

        # Check expected output in stdout
        assert "No site specified, parsing all available sites with crawled content" in result.stdout
        assert f"Content directory not found: {DEFAULT_CONTENT_DIR}" in result.stdout

    @patch("tapio.cli.ConfigManager")
    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("os.walk")
    def test_parse_command_no_site_no_crawled_content(
        self,
        mock_walk,
        mock_isdir,
        mock_listdir,
        mock_exists,
        mock_config_manager,
        runner,
    ):
        """Test the parse command when no site is specified and no crawled content is found."""
        # Setup mocks - directory exists but contains no HTML files
        mock_exists.return_value = True
        mock_listdir.return_value = ["url_mappings.json", "empty_dir"]
        mock_isdir.side_effect = lambda path: path.endswith("empty_dir")

        # Mock os.walk to return no HTML files
        mock_walk.return_value = [("/content/crawled/empty_dir", [], ["text_file.txt"])]

        mock_config_instance = MagicMock()
        mock_config_instance.list_available_sites.return_value = ["migri", "kela"]
        mock_config_manager.return_value = mock_config_instance

        # Run the command without --site parameter
        result = runner.invoke(app, ["parse"])

        # Check that the command exited with error code
        assert result.exit_code == 1

        # Check expected output in stdout
        assert "No site specified, parsing all available sites with crawled content" in result.stdout
        assert "No crawled content found to parse" in result.stdout

    @patch("tapio.cli.ConfigManager")
    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("os.walk")
    def test_parse_command_no_site_no_matching_configs(
        self,
        mock_walk,
        mock_isdir,
        mock_listdir,
        mock_exists,
        mock_config_manager,
        runner,
    ):
        """Test the parse command when crawled content exists but no site configs match."""
        # Setup mocks for new directory structure - content/site_name/crawled/
        mock_exists.side_effect = lambda path: (
            path == DEFAULT_CONTENT_DIR or "unknown_site/crawled" in path or "another_unknown/crawled" in path
        )
        mock_listdir.return_value = ["unknown_site", "another_unknown"]
        mock_isdir.side_effect = lambda path: not path.endswith(".json")

        # Mock os.walk to return HTML files for unknown sites
        def mock_walk_side_effect(path):
            if "unknown_site/crawled" in path:
                return [("/content/unknown_site/crawled", [], ["page1.html"])]
            elif "another_unknown/crawled" in path:
                return [("/content/another_unknown/crawled", [], ["page2.html"])]
            return []

        mock_walk.side_effect = mock_walk_side_effect

        # Set up mock config manager with sites that don't match the crawled sites
        mock_config_instance = MagicMock()
        mock_config_instance.list_available_sites.return_value = ["migri", "kela"]

        mock_config_manager.return_value = mock_config_instance

        # Run the command without --site parameter
        result = runner.invoke(app, ["parse"])

        # Check that the command exited with error code
        assert result.exit_code == 1

        # Check expected output in stdout
        assert "No site specified, parsing all available sites with crawled content" in result.stdout
        expected_sites = "Crawled sites: unknown_site, another_unknown"
        assert expected_sites in result.stdout
        assert "No site configurations found matching crawled content" in result.stdout
        assert "Available sites: migri, kela" in result.stdout

    @patch("tapio.cli.ConfigManager")
    @patch("tapio.cli.Parser")
    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("os.walk")
    def test_parse_command_no_site_partial_match(
        self,
        mock_walk,
        mock_isdir,
        mock_listdir,
        mock_exists,
        mock_parser,
        mock_config_manager,
        runner,
    ):
        """Test the parse command when only some crawled sites have matching site configs."""
        # Setup mocks for directory structure - new structure uses content/site_name/crawled/
        mock_exists.side_effect = lambda path: (
            path == DEFAULT_CONTENT_DIR
            or "migri/crawled" in path
            or "unknown/crawled" in path
            or "kela/crawled" in path
        )
        mock_listdir.return_value = ["migri", "unknown", "kela"]
        mock_isdir.side_effect = lambda path: not path.endswith(".json")

        # Mock os.walk to return HTML files
        def mock_walk_side_effect(path):
            if "migri/crawled" in path:
                return [("/content/migri/crawled", [], ["page1.html"])]
            elif "kela/crawled" in path:
                return [("/content/kela/crawled", [], ["page2.html"])]
            elif "unknown/crawled" in path:
                return [("/content/unknown/crawled", [], ["page3.html"])]
            return []

        mock_walk.side_effect = mock_walk_side_effect

        # Set up mock parser instances for 2 matching sites
        mock_parser_instances = []
        for i in range(2):
            mock_instance = MagicMock()
            mock_instance.parse_all.return_value = [f"file{i + 1}"]
            mock_parser_instances.append(mock_instance)

        mock_parser.side_effect = mock_parser_instances

        # Set up mock config manager - only migri and kela are available, unknown is not
        mock_config_instance = MagicMock()
        mock_config_instance.list_available_sites.return_value = ["migri", "kela", "vero"]

        mock_config_manager.return_value = mock_config_instance

        # Run the command without --site parameter
        result = runner.invoke(app, ["parse"])

        # Check that the command ran successfully
        assert result.exit_code == 0

        # Check that only 2 parsers were created (for migri and kela)
        assert mock_parser.call_count == 2

        # Check expected output in stdout
        assert "Found crawled content for sites: migri, unknown, kela" in result.stdout
        assert "Parsing sites: migri, kela" in result.stdout
        assert "All parsing completed! Processed 2 files total." in result.stdout
        assert "Parsed 2 sites: migri, kela" in result.stdout

    @patch("tapio.cli.ConfigManager")
    @patch("tapio.cli.Parser")
    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("os.walk")
    def test_parse_command_no_site_with_exception(
        self,
        mock_walk,
        mock_isdir,
        mock_listdir,
        mock_exists,
        mock_parser,
        mock_config_manager,
        runner,
    ):
        """Test handling of exceptions in parse command when parsing all sites."""
        # Setup mocks for directory structure - new structure uses content/site_name/crawled/
        mock_exists.side_effect = lambda path: (path == DEFAULT_CONTENT_DIR or "migri/crawled" in path)
        mock_listdir.return_value = ["migri"]
        mock_isdir.side_effect = lambda path: not path.endswith(".json")

        def mock_walk_side_effect(path):
            if "migri/crawled" in path:
                return [("/content/migri/crawled", [], ["page.html"])]
            return []

        mock_walk.side_effect = mock_walk_side_effect

        # Set up mock parser that raises an exception
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_all.side_effect = Exception("Test parsing error")
        mock_parser.return_value = mock_parser_instance

        # Set up mock config manager
        mock_config_instance = MagicMock()
        mock_config_instance.list_available_sites.return_value = ["migri"]

        mock_config_manager.return_value = mock_config_instance

        # Run the command without --site parameter
        result = runner.invoke(app, ["parse"])

        # Check that the command exited with error code
        assert result.exit_code == 1

        # Check expected output in stdout
        assert "No site specified, parsing all available sites with crawled content" in result.stdout
        assert "Error during parsing: Test parsing error" in result.stdout

    @patch("tapio.cli.MarkdownVectorizer")
    def test_vectorize_command_with_nonexistent_site(self, mock_vectorizer, runner):
        """Test the vectorize command with a non-existent site."""
        # Mock os.path.exists to return False for the site directory
        with patch("tapio.cli.os.path.exists", return_value=False):
            # Run the command with non-existent site
            result = runner.invoke(app, ["vectorize", "--site", "nonexistent"])

        # Check that the command exited with error code
        assert result.exit_code == 1

        # Check expected output in stdout
        assert "No parsed content found for site: nonexistent" in result.stdout
        assert "content/nonexistent/parsed" in result.stdout

        # Verify that vectorizer was not called
        mock_vectorizer.assert_not_called()
