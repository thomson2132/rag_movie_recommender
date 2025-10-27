import os
import yaml
import logging
from dotenv import load_dotenv
import torch


def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file with UTF-8 encoding."""
    with open(config_path, 'r', encoding='utf-8') as f:  # ← Added encoding='utf-8'
        config = yaml.safe_load(f)
    return config


def setup_environment():
    """Load environment variables and setup CUDA."""
    load_dotenv()

    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠ CUDA not available. Using CPU")

    return device


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log', encoding='utf-8'),  # ← Added encoding='utf-8'
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_groq_api_key():
    """Retrieve Groq API key from environment."""
    load_dotenv()
    api_key = os.getenv('GROQ_API_KEY')

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in environment variables.\n"
            "Please create a .env file with: GROQ_API_KEY=gsk_your_key_here"
        )

    if not api_key.startswith('gsk_'):
        raise ValueError(
            "Invalid GROQ_API_KEY format. Key should start with 'gsk_'\n"
            "Get your API key from: https://console.groq.com/"
        )

    return api_key
