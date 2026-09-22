"""
Hugging Face Spaces entry point.

Spaces looks for `app.py` and supplies its own host and port, so this module
does nothing but import the Gradio app and launch it with defaults.

For local use, prefer `run_app.py`, which pins the app to 127.0.0.1. Running
`food_agent.py` directly launches with share=True and opens a public tunnel.

The GROQ_API_KEY comes from the Space's Secrets (Settings -> Variables and
secrets). groq_client.py reads the real environment before the .env file, so
nothing here needs to change between local and Spaces.
"""

from food_agent import demo

demo.launch()
