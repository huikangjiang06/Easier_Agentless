import os
import time
from typing import Dict, Union

import anthropic
import openai
import tiktoken

try:
    from google import genai
except ImportError:
    genai = None


def num_tokens_from_messages(message, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(message, list):
        # use last message.
        num_tokens = len(encoding.encode(message[0]["content"]))
    else:
        num_tokens = len(encoding.encode(message))
    return num_tokens


def create_chatgpt_config(
    message: Union[str, list],
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-3.5-turbo",
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [{"role": "system", "content": system_message}] + message,
        }
    else:
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ],
        }
    return config


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


def request_chatgpt_engine(config, logger, base_url=None, max_retries=40, timeout=100):
    ret = None
    retries = 0

    client = openai.OpenAI(base_url=base_url)

    while ret is None and retries < max_retries:
        try:
            # Attempt to get the completion
            logger.info("Creating API request")

            ret = client.chat.completions.create(**config)

        except openai.OpenAIError as e:
            if isinstance(e, openai.BadRequestError):
                logger.info("Request invalid")
                print(e)
                logger.info(e)
                raise Exception("Invalid API Request")
            elif isinstance(e, openai.RateLimitError):
                print("Rate limit exceeded. Waiting...")
                logger.info("Rate limit exceeded. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            elif isinstance(e, openai.APIConnectionError):
                print("API connection error. Waiting...")
                logger.info("API connection error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            else:
                print("Unknown error. Waiting...")
                logger.info("Unknown error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(1)

        retries += 1

    logger.info(f"API response {ret}")
    return ret


def create_anthropic_config(
    message: str,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "claude-2.1",
    tools: list = None,
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": message,
        }
    else:
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": message}]},
            ],
        }

    if tools:
        config["tools"] = tools

    return config


def request_anthropic_engine(
    config, logger, max_retries=40, timeout=500, prompt_cache=False
):
    ret = None
    retries = 0

    client = anthropic.Anthropic()

    while ret is None and retries < max_retries:
        try:
            start_time = time.time()
            if prompt_cache:
                # following best practice to cache mainly the reused content at the beginning
                # this includes any tools, system messages (which is already handled since we try to cache the first message)
                config["messages"][0]["content"][0]["cache_control"] = {
                    "type": "ephemeral"
                }
                ret = client.beta.prompt_caching.messages.create(**config)
            else:
                ret = client.messages.create(**config)
        except Exception as e:
            logger.error("Unknown error. Waiting...", exc_info=True)
            # Check if the timeout has been exceeded
            if time.time() - start_time >= timeout:
                logger.warning("Request timed out. Retrying...")
            else:
                logger.warning("Retrying after an unknown error...")
            time.sleep(10 * retries)
        retries += 1

    return ret


def create_vertexai_config(
    message: Union[str, list],
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "gemini-2.5-pro",
    project: str = None,
    location: str = None,
) -> Dict:
    """Create configuration for Vertex AI Gemini API requests."""
    # Use environment variables if project/location not provided
    if project is None:
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "triangulate-396717")
    if location is None:
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    if isinstance(message, list):
        # Convert list of messages to content format
        contents = []
        for msg in message:
            if isinstance(msg, dict):
                if msg.get("role") == "user":
                    contents.append(msg.get("content", ""))
                elif msg.get("role") == "system":
                    # System message can be prepended
                    contents.insert(0, msg.get("content", ""))
        final_content = "\n".join(str(c) for c in contents)
    else:
        final_content = message
    
    config = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "batch_size": batch_size,
        "message": final_content,
        "project": project,
        "location": location,
    }
    return config


def request_vertexai_engine(
    config, logger, max_retries=40, timeout=500
):
    """Request completion from Vertex AI Gemini."""
    if genai is None:
        raise ImportError(
            "google-genai is not installed. Please install it with: pip install google-genai"
        )
    
    ret = None
    retries = 0

    # Initialize Vertex AI client with credentials
    client = genai.Client(
        vertexai=True,
        project=config.get("project"),
        location=config.get("location"),
    )

    while ret is None and retries < max_retries:
        try:
            logger.info("Creating Vertex AI API request")
            
            # Make the API call
            response = client.models.generate_content(
                model=config["model"],
                contents=config["message"],
            )
            
            # Wrap response in a simple object that mimics the expected interface
            ret = VertexAIResponse(response)
            
        except Exception as e:
            logger.error(f"Vertex AI request error: {e}", exc_info=True)
            if retries < max_retries - 1:
                wait_time = min(5 * (retries + 1), 60)  # Exponential backoff, max 60s
                logger.warning(f"Retrying after {wait_time}s...")
                time.sleep(wait_time)
        
        retries += 1

    logger.info(f"Vertex AI response: {ret}")
    return ret


class VertexAIResponse:
    """Wrapper to standardize Vertex AI response format."""
    
    def __init__(self, response):
        self.response = response
        self.text = response.text if hasattr(response, 'text') else str(response)
        
        # Estimate token usage (Vertex AI may not provide exact counts)
        # We'll use a simple heuristic: ~1 token per 4 characters
        self.usage = VertexAIUsage(
            input_tokens=0,  # Not directly available from Vertex AI
            output_tokens=len(self.text) // 4,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
    
    def to_dict(self):
        return {
            "text": self.text,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
            }
        }


class VertexAIUsage:
    """Usage statistics wrapper for Vertex AI responses."""
    
    def __init__(self, input_tokens=0, output_tokens=0, 
                 cache_creation_input_tokens=0, cache_read_input_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
