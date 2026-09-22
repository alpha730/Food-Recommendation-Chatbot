"""
Deployment entry point (Render, Hugging Face Spaces, Railway, Fly, ...).

Hosts bind the app themselves, so this module only imports the Gradio app and
launches it on the interface and port the platform asks for:

    server_name="0.0.0.0"   reachable from outside the container, not just
                            localhost - a host that cannot reach the app will
                            report the deploy as failed
    server_port=$PORT       the port the platform assigns; 7860 is Gradio's
                            default and what Spaces uses

For local use, prefer `run_app.py`, which pins the app to 127.0.0.1. Running
`food_agent.py` directly launches with share=True and opens a public tunnel.

The GROQ_API_KEY comes from the platform's secret store (on Render: Environment
-> Environment Variables; on Spaces: Settings -> Variables and secrets).
groq_client.py reads the real environment before the .env file, so nothing here
changes between local and deployed runs.
"""

import os

from food_agent import demo

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
)
