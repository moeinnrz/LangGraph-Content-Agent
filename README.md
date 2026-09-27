# LangGraph Content Agent

An agentic content-generation workflow built with LangGraph. The system generates content, reviews it with an LLM, conditionally revises it, and returns a final version.

## Workflow

```text
                    +----------------+
                    | Generate Draft |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    |  AI Reviewer    |
                    +-------+--------+
                            |
                    +-------+--------+
                    |                |
                Score >= 8       Needs work
                    |                |
                    v                v
               +---------+    +-------------+
               | Finalize |    |    Revise   |
               +---------+    +------+------+
                                      |
                                      +------> Review
```

## Features

- LangGraph StateGraph workflow
- Stateful agent execution
- LLM-based content generation
- Automated content review
- Quality scoring from 1 to 10
- Conditional workflow routing
- Automatic revision loop
- Configurable maximum revisions
- Configurable passing score
- Colorized CLI
- Environment-based configuration

## Requirements

- Python 3.10+
- A Groq API key

## Installation

```bash
git clone https://github.com/moeinnrz/LangGraph-Content-Agent.git
cd LangGraph-Content-Agent

python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Create `.env` from `.env.example`:

```env
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
MAX_REVISIONS=3
PASS_SCORE=8
```

Run:

```bash
python main.py
```

## State

The workflow maintains:

- `topic`
- `draft`
- `feedback`
- `score`
- `revision_count`
- `final_content`

## How It Works

1. The user provides a topic.
2. The generator creates an initial draft.
3. The reviewer evaluates the draft and assigns a score.
4. If the score reaches the configured threshold, the workflow finalizes.
5. Otherwise, the revision node improves the draft.
6. The revised draft is reviewed again.
7. The process stops after the score passes or the maximum revision count is reached.

## Security

Never commit `.env` or real API keys.

## License

This project is provided for educational and portfolio purposes.
