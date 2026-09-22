# Food Recommendation Chatbot

A multi-agent food recommendation system running entirely on [Groq](https://console.groq.com/).
Six specialized LLM agents collaborate over a shared state to turn a plain-English request
("somewhere vegetarian in Silver Lake for a date night") into structured restaurant and
recipe recommendations, served through a Gradio chat interface.

```
You: I want healthy Japanese food in LA, no shellfish

  Phase 1  User Profile Generator   -> preferences, restrictions, dining patterns
  Phase 2  RAG Retriever            -> candidate restaurants + recipes
  Phase 3  Food Trend Analyst   ┐
           Food Style Expert    ├─ run in parallel
           Nutrition Expert     ┘
  Phase 4  Recommendation Expert    -> ranked, explained recommendations
```

## Quick start

Requires Python 3.11+ and a free Groq API key from https://console.groq.com/keys.

```bash
git clone https://github.com/alpha730/Food-Recommendation-Chatbot.git
cd Food-Recommendation-Chatbot

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env               # Windows: copy .env.example .env
# then open .env and set GROQ_API_KEY=gsk_...

python run_app.py
```

The app opens at http://127.0.0.1:7860.

Running `python food_agent.py` directly also works, but note it launches with
`share=True`, which creates a **public Gradio tunnel URL** anyone can open. `run_app.py`
pins it to localhost instead. Prefer `run_app.py` unless you actually want to share the
running app.

## Configuration

Everything lives in a single `.env` file next to `groq_client.py`. Nothing else needs editing.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | yes | — | Your Groq key |
| `GROQ_MODEL` | no | `qwen/qwen3.8-27b` | Model for the six agents |
| `GROQ_CHAT_MODEL` | no | `openai/gpt-oss-20b` | Model for intent classification and preference extraction |

Real environment variables win over the `.env` file, so `export GROQ_API_KEY=...` (or `set` on
Windows) still overrides it in CI or a shell. `.env` is gitignored — only `.env.example` is
committed, and it contains a placeholder, not a key.

If no key is found, the app still starts but prints a warning and every model call fails.

## The interface

| Tab | What it does |
|---|---|
| 💬 **Chat** | Describe what you want; the agents run and return recommendations |
| ➕ **Add Restaurant** | Add a restaurant to the in-session database |
| ➕ **Add Recipe** | Add a recipe to the in-session database |
| ℹ️ **About** | System overview |

Incoming messages are first classified into one of `restaurant`, `recipe`, `both`,
`clarification`, or `database`. Only the first three trigger the full workflow — the other two
return help text immediately, so casual questions don't burn six LLM calls.

## How it works

`food_agent.py` is the whole application, in labelled sections:

- **Agent configs** — each of the six agents is a `role` / `goal` / `backstory` dict, rendered
  into a system prompt by `create_agent_prompt`.
- **Shared state** — a single dict threaded through every node, holding the user profile,
  retrieved candidates, the three analyses, and the final recommendations.
- **Workflow nodes** — one function per agent, each reading from and writing to that state.
- **`run_workflow`** — the four phases above. Phase 3's three agents are independent, so they
  run concurrently on a `ThreadPoolExecutor`, each with its own copy of the state; results are
  merged afterward.
- **Integration layer** — wraps `call_agent` to tolerate fenced JSON in model output, and
  connects the Gradio chat to the real workflow.

## Data

Three datasets ship with the repo:

| File | Contents |
|---|---|
| `California-Culinary-Map.txt` | 210 California restaurants as prose descriptions |
| `Recipes.json` | 109 recipes with ingredients, timings, and directions |
| `Synthetic-User-Reviews.json` | 10 synthetic user reviews |

The matching recipe image set (~205 MB) is **not** in the repo — it exceeds GitHub's 100 MB
per-file limit and is gitignored.

## Repository layout

```
run_app.py              Launcher — start here
food_agent.py           The complete application (agents, workflow, Gradio UI)
groq_client.py          Single source of truth for the API key and model names
requirements.txt        Dependencies for the app

data.py                 Standalone: parse the restaurant text file
exercise3.py            Standalone: structure restaurant prose into JSON via Groq

m1l1_*.py  m1l2_*.py    Course lesson modules (reference)
m2l1_*.py  m2l2_*.py  m2l3_*.py
m3l1_*.py  m3l2_*.py  m3l3_*.py
lib.py                  Helper used by the Module 1 lessons
```

### A note on the lesson modules

The `m1*` / `m2*` / `m3*` files were extracted verbatim from the course notebooks this project
was built from, and are kept as reference for how the system was assembled. They are **not**
wired into the app and **will not run from a clean install** — they variously need
`torch`, `sentence-transformers`, `transformers`, `langchain-chroma`, and `ibm-watsonx-ai`
(none of which are in `requirements.txt`), and some need the image set that isn't in the repo.

The three Module 3 notebooks are the exception in that their content *is* live — but as
rewritten code inside `food_agent.py`, where every OpenAI dependency (`openai`,
`langchain_openai.ChatOpenAI`, `OPENAI_API_KEY`) was replaced with Groq equivalents. The agent
logic, prompts, and interface are otherwise unchanged from the notebooks.

Install the app dependencies only, and everything under **Quick start** works.
