# Academic Essay Generator

A local multi-agent system for generating PhD-level academic essays using LangGraph for agent orchestration and Ollama for local LLM inference. The system processes literature PDFs, generates structured essays, and includes a review/editing pipeline with automatic revision cycles.

## Features

- **6 Specialized Agents**: Research, Outline, Writer, Citation, Review, and Editor agents working in a coordinated pipeline
- **Local LLM Processing**: Uses Ollama for privacy and offline operation
- **PDF Literature Processing**: Automatically extracts and chunks text from academic PDFs
- **Structured Output**: Generates well-organized essays with proper citations (APA format)
- **Review Loop**: Automatic quality assessment with revision cycles (up to 2 revisions)
- **Academic Standards**: Designed for PhD-level academic writing with proper tone and conventions

## Requirements

- Python 3.11 or higher
- Ollama installed and running locally
- One of the following models available in Ollama:
  - `llama3.1:8b-instruct-q4_K_M` (default)
  - `qwen2.5:7b` (alternative)
  - Any compatible instruction-tuned model

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd academic-essay-generator
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and set up Ollama:**
   - Download from [https://ollama.ai](https://ollama.ai)
   - Install the model:
     ```bash
     ollama pull llama3.1:8b-instruct-q4_K_M
     ```
   - Or use an alternative model:
     ```bash
     ollama pull qwen2.5:7b
     ```

4. **Verify Ollama is running:**
   ```bash
   ollama list
   ```

5. **Set up environment variables (optional):**

   The system supports optional Langfuse tracking for observability. If you want to enable tracking, set the following environment variables:

   ```bash
   export LANGFUSE_PUBLIC_KEY="your-public-key"
   export LANGFUSE_SECRET_KEY="your-secret-key"
   ```

   Or create a `.env` file in the project root:
   ```
   LANGFUSE_PUBLIC_KEY=your-public-key
   LANGFUSE_SECRET_KEY=your-secret-key
   ```

   **Note:** Tracking is optional. If these environment variables are not set, the system will run without tracking. You can get your Langfuse credentials from [https://cloud.langfuse.com](https://cloud.langfuse.com) or disable tracking in `config.yaml` by setting `tracking.enabled: false`.

## Usage

### Basic Command

```bash
python main.py \
  --topic "The impact of transformer architectures on NLP" \
  --criteria "criteria.txt" \
  --literature ./inputs/ \
  --output ./outputs/essay.md
```

### Command Arguments

- `--topic` / `-t`: The essay topic (required)
- `--criteria` / `-c`: Path to a text file containing evaluation criteria (required)
- `--literature` / `-l`: Path to directory containing PDF files (required)
- `--output` / `-o`: Output path for the generated essay (required)
- `--config`: Optional path to config.yaml (default: `./config.yaml`)

### Example Workflow

1. **Prepare your literature:**
   - Place PDF files in the `inputs/` directory
   - Ensure PDFs are readable and contain relevant academic content

2. **Create criteria file:**
   - Create a text file (e.g., `criteria.txt`) with your evaluation criteria
   - Example:
     ```
     The essay should:
     - Demonstrate deep understanding of transformer architectures
     - Compare different transformer models
     - Discuss applications in NLP tasks
     - Include critical analysis of limitations
     - Be approximately 5000-6000 words
     ```

3. **Run the generator:**
   ```bash
      python main.py \
        --topic "Smarter Shields: How AI is Transforming Firewall Policy Automation in Enterprise Networks" \
        --criteria criteria.txt \
        --literature ./inputs/ \
        --output ./outputs/essay.md
   ```

4. **Monitor progress:**
   - The system will display progress for each agent
   - Review scores and revision cycles are shown
   - Final essay is saved to the specified output path

## Configuration

Edit `config.yaml` to customize the system:

```yaml
# Ollama Settings
ollama:
  model: "llama3.1:8b-instruct-q4_K_M"  # Change model here
  base_url: "http://localhost:11434"
  timeout: 300

# Document Processing
chunking:
  chunk_size: 1000      # Approximate tokens per chunk
  chunk_overlap: 100    # Overlap between chunks

# Review Settings
review:
  threshold: 0.7        # Minimum score to pass (0.0-1.0)
  max_revision_cycles: 2  # Maximum revision attempts

# Tracking Settings (optional)
tracking:
  enabled: true         # Enable/disable Langfuse tracking
  provider: "langfuse"
  langfuse:
    public_key: "${LANGFUSE_PUBLIC_KEY}"  # From environment variable
    secret_key: "${LANGFUSE_SECRET_KEY}"  # From environment variable
    host: "https://cloud.langfuse.com"
```

## Project Structure

```
academic-essay-generator/
├── src/
│   ├── agents/          # Agent implementations
│   │   ├── research_agent.py
│   │   ├── outline_agent.py
│   │   ├── writer_agent.py
│   │   ├── citation_agent.py
│   │   ├── review_agent.py
│   │   └── editor_agent.py
│   ├── graph/           # LangGraph workflow
│   │   └── workflow.py
│   ├── loaders/         # Document processing
│   │   └── pdf_loader.py
│   ├── state/           # State management
│   │   └── state.py
│   └── utils/           # Utilities
│       ├── ollama_client.py
│       └── prompts.py
├── inputs/              # Place PDF files here
├── outputs/             # Generated essays saved here
├── config.yaml          # Configuration file
├── main.py              # CLI entry point
└── requirements.txt     # Python dependencies
```

## How It Works

### Agent Pipeline

1. **ResearchAgent**: Analyzes uploaded literature PDFs, extracts key arguments, quotes, and themes
2. **OutlineAgent**: Creates a structured essay outline based on topic and evaluation criteria
3. **WriterAgent**: Generates essay sections one at a time following the outline
4. **CitationAgent**: Identifies citation needs and formats them in APA style
5. **ReviewAgent**: Evaluates the draft against criteria, provides feedback and scores (0.0-1.0)
6. **EditorAgent**: Performs final polish, ensures coherence, and formats as Markdown

### Review Loop

- If the review score is below the threshold (default: 0.7) and revision cycles remain, the system routes back to WriterAgent
- Maximum 2 revision cycles to prevent infinite loops
- After revisions or if score is acceptable, the essay proceeds to EditorAgent

### State Management

The system uses a Pydantic model (`EssayState`) to manage shared state across agents:
- Topic and criteria
- Literature chunks
- Research notes (structured JSON)
- Outline (structured JSON)
- Generated sections
- Citations
- Review feedback and scores
- Final essay

## Output Format

The generated essay is saved as Markdown with:
- Title and introduction
- Structured sections with headings
- Inline citations in format `[Author, Year]`
- Bibliography section at the end
- Academic formatting and tone

## Troubleshooting

### Ollama Connection Issues

If you see connection errors:
1. Verify Ollama is running: `ollama list`
2. Check the base URL in `config.yaml` matches your Ollama instance
3. Ensure the specified model is installed: `ollama pull <model-name>`

### PDF Processing Issues

- Ensure PDFs are not password-protected
- Check that PDFs contain extractable text (not just images)
- Verify PDFs are in the correct directory

### Low Quality Output

- Try a different model (e.g., larger model or different architecture)
- Adjust the review threshold in `config.yaml`
- Provide more detailed criteria
- Ensure literature PDFs are relevant to the topic

## Limitations

- Processing happens in-memory (no vector databases)
- Target output is 12-15 pages (~5000-6000 words)
- Maximum 2 revision cycles
- APA citation format only
- Requires local Ollama installation

## License

This project is provided as-is for academic and research purposes.

## Contributing

This is a focused implementation for academic essay generation. For improvements:
1. Test with different models
2. Adjust prompts in `src/utils/prompts.py`
3. Modify agent logic in `src/agents/`
4. Customize workflow in `src/graph/workflow.py`

