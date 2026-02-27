# Easier Agentless

> Fork of [OpenAutoCoder/Agentless](https://github.com/OpenAutoCoder/Agentless) with simplified execution and multi-backend support.

## What's New

This fork adds two major improvements to the original Agentless repository:

### 1. **Aggregated Execution**
The original Agentless requires running 16 separate commands with different arguments for each step. This fork provides:
- **Single execution script** (`run.sh`) that runs all steps
- **Config-driven workflow** (`config.toml`) for setting target instances and LLM parameters
- **Simpler workflow** - edit config once, run one command

### 2. **Multi-Backend Support**
Added support for multiple LLM providers beyond OpenAI:
- **OpenAI** (original): GPT-4o, GPT-4o-mini
- **Anthropic**: Claude 3.5 Sonnet
- **DeepSeek**: DeepSeek Coder
- **Vertex AI**: Gemini 2.5 Pro, Gemini 1.5 Pro

## Quick Start

```bash
# Setup
conda create -n agentless python=3.11 
conda activate agentless
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Configure targets and backend
cat > config.toml << EOF
target_ids = ["django__django-10914"]
backend = "openai"
model = "gpt-4o-2024-05-13"
EOF

# Set API key
export OPENAI_API_KEY="sk-..."

# Run all 16 steps
./run.sh
```

Results are saved to `results/swe-bench-lite/`.

## Configuration

Edit `config.toml` to customize your run:

```toml
# Instance IDs to process
target_ids = [
  "django__django-10914",
  "sphinx-doc__sphinx-8282"
]

# LLM backend: "openai", "anthropic", "deepseek", or "vertexai"
backend = "openai"

# Model name (must match backend)
model = "gpt-4o-2024-05-13"

# Optional: API key (or use environment variable)
# openai_api_key = "sk-..."
```

### Backend Configuration

| Backend | Environment Variable | Example Model |
|---------|---------------------|---------------|
| OpenAI | `OPENAI_API_KEY` | gpt-4o-2024-05-13 |
| Anthropic | `ANTHROPIC_API_KEY` | claude-3-5-sonnet-20241022 |
| DeepSeek | `DEEPSEEK_API_KEY` | deepseek-coder |
| Vertex AI | `GOOGLE_CLOUD_PROJECT`<br>`GOOGLE_CLOUD_LOCATION` | gemini-2.5-pro |

**Vertex AI setup:**
```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-project"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

## Why This Fork?

**Problem:** The original workflow requires:
- Running 16 different commands manually
- Remembering complex argument patterns for each step
- Manually managing intermediate results between steps
- OpenAI-only support

**Solution:** This fork provides:
- Single command execution with `run.sh`
- All configuration in one place (`config.toml`)
- Automatic chaining of all pipeline steps
- Support for 4 LLM providers (OpenAI, Anthropic, DeepSeek, Vertex AI)

## Changes Summary

### Modified Files
- `agentless/util/api_requests.py` - Added Vertex AI API integration
- `agentless/util/model.py` - Added VertexAIGeminiDecoder class
- `agentless/fl/localize.py` - Added backend/model arguments, vertexai support
- `agentless/fl/combine.py` - Added incremental processing (skip existing)
- `agentless/repair/repair.py` - Added backend/model arguments
- `agentless/test/*.py` - Added backend/model arguments
- `requirements.txt` - Added google-genai, tomli

### New Files
- `run.sh` - Aggregated execution script
- `config.toml` - Configuration file (example)

## Original Documentation

For detailed information about the Agentless approach, benchmark results, and step-by-step manual execution:
- [Original Agentless README](https://github.com/OpenAutoCoder/Agentless)
- [SWE-bench Step-by-Step Guide](README_swebench.md) (manual execution)

## Citation

```bibtex
@article{agentless,
  author    = {Xia, Chunqiu Steven and Deng, Yinlin and Dunn, Soren and Zhang, Lingming},
  title     = {Agentless: Demystifying LLM-based Software Engineering Agents},
  year      = {2024},
  journal   = {arXiv preprint},
}
```
