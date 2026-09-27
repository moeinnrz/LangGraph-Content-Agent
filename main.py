import os
from typing import TypedDict

from colorama import Fore, Style, init
from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import END, StateGraph

load_dotenv()
init(autoreset=True)

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "3"))
PASS_SCORE = int(os.getenv("PASS_SCORE", "8"))

if not API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")

client = Groq(api_key=API_KEY)


class AgentState(TypedDict):
    topic: str
    draft: str
    feedback: str
    score: int
    revision_count: int
    final_content: str


def call_llm(prompt: str, temperature: float = 0.5) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def generate(state: AgentState):
    prompt = f"""
Create a high-quality article about this topic:

{state["topic"]}

Requirements:
- Clear structure
- Useful information
- Professional tone
- Concise but substantial writing
- Use headings when appropriate
"""

    draft = call_llm(prompt, temperature=0.7)

    return {
        "draft": draft,
        "feedback": "",
        "score": 0,
    }


def review(state: AgentState):
    prompt = f"""
Review the following article.

Topic:
{state["topic"]}

Article:
{state["draft"]}

Evaluate:
- Accuracy and relevance
- Structure
- Clarity
- Usefulness
- Writing quality

Return exactly this format:

SCORE: <number from 1 to 10>
FEEDBACK: <specific improvements>
"""

    review_text = call_llm(prompt, temperature=0.2)

    score = 0
    feedback = review_text

    for line in review_text.splitlines():
        if line.upper().startswith("SCORE:"):
            try:
                score = int(line.split(":", 1)[1].strip())
            except ValueError:
                score = 0
        elif line.upper().startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()

    score = max(0, min(score, 10))

    return {
        "score": score,
        "feedback": feedback,
    }


def revise(state: AgentState):
    prompt = f"""
Improve the article below using the review feedback.

Topic:
{state["topic"]}

Current article:
{state["draft"]}

Reviewer score:
{state["score"]}/10

Reviewer feedback:
{state["feedback"]}

Return only the revised article.
"""

    revised = call_llm(prompt, temperature=0.5)

    return {
        "draft": revised,
        "revision_count": state["revision_count"] + 1,
    }


def finalize(state: AgentState):
    return {"final_content": state["draft"]}


def route_after_review(state: AgentState):
    if state["score"] >= PASS_SCORE:
        return "finalize"

    if state["revision_count"] >= MAX_REVISIONS:
        return "finalize"

    return "revise"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("generate", generate)
    graph.add_node("review", review)
    graph.add_node("revise", revise)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "review")

    graph.add_conditional_edges(
        "review",
        route_after_review,
        {
            "revise": "revise",
            "finalize": "finalize",
        },
    )

    graph.add_edge("revise", "review")
    graph.add_edge("finalize", END)

    return graph.compile()


def run_agent(topic: str):
    app = build_graph()

    initial_state: AgentState = {
        "topic": topic,
        "draft": "",
        "feedback": "",
        "score": 0,
        "revision_count": 0,
        "final_content": "",
    }

    return app.invoke(initial_state)


def main():
    print(Fore.CYAN + Style.BRIGHT + "\n" + "=" * 58)
    print("              LANGGRAPH CONTENT AGENT")
    print("=" * 58 + Style.RESET_ALL)

    print(Fore.WHITE + "Enter a topic and let the agent generate, review, and revise it.")
    print(Fore.WHITE + "Type 'exit' or 'quit' to stop.\n")

    while True:
        topic = input(Fore.BLUE + "Topic > " + Style.RESET_ALL).strip()

        if not topic:
            continue

        if topic.lower() in {"exit", "quit"}:
            print(Fore.CYAN + "Goodbye!")
            break

        try:
            print(Fore.YELLOW + "Running LangGraph workflow...")

            result = run_agent(topic)

            print(
                Fore.GREEN
                + f"Final score: {result['score']}/10 | "
                f"Revisions: {result['revision_count']}"
            )

            print(Fore.GREEN + "\n" + result["final_content"] + "\n")

        except Exception as exc:
            print(Fore.RED + f"Error: {exc}\n")


if __name__ == "__main__":
    main()
