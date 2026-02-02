"""
Configuration module for loading API keys from environment variables or .env file.
This keeps API keys out of the source code.
"""

import os
from pathlib import Path

# Try to load python-dotenv if available
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, will use environment variables only
    pass


def get_api_keys():
    """
    Get API keys from environment variables.
    
    Returns:
        tuple: (tmdb_api_key, openai_api_key)
    
    Raises:
        ValueError: If required API keys are not set
    """
    tmdb_key = os.getenv("TMDB_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not tmdb_key:
        raise ValueError(
            "TMDB_API_KEY not found. Please set it as an environment variable "
            "or create a .env file with TMDB_API_KEY=your_key"
        )
    
    if not openai_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it as an environment variable "
            "or create a .env file with OPENAI_API_KEY=your_key"
        )
    
    return tmdb_key, openai_key


def get_tmdb_api_key():
    """Get TMDB API key from environment."""
    key = os.getenv("TMDB_API_KEY")
    if not key:
        raise ValueError(
            "TMDB_API_KEY not found. Please set it as an environment variable "
            "or create a .env file with TMDB_API_KEY=your_key"
        )
    return key


def get_openai_api_key():
    """Get OpenAI API key from environment."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it as an environment variable "
            "or create a .env file with OPENAI_API_KEY=your_key"
        )
    return key


def get_anthropic_api_key():
    """Get Anthropic (Claude) API key from environment. Used for 'need' field generation."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found. Set it in your environment or .env for need-field generation."
        )
    return key


def get_intent_model() -> str:
    """
    Model used for query intent parsing. Use a capable model for accurate intent extraction.
    Default: gpt-4o. Override via INTENT_MODEL env var (e.g. gpt-4o, gpt-4-turbo, gpt-4o-mini).
    """
    return os.getenv("INTENT_MODEL", "gpt-4o").strip() or "gpt-4o"
