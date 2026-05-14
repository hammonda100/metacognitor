import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MindConfig
from web_app import app, config
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous Learning Mind")
    parser.add_argument("--host", default=config.host, help="Host to bind")
    parser.add_argument("--port", type=int, default=config.port, help="Port to bind")
    args = parser.parse_args()

    print(f"╔════════════════════════════════════════════╗")
    print(f"║   Continuous Learning Mind v2.0            ║")
    print(f"╠════════════════════════════════════════════╣")
    print(f"  LLM Provider: {config.llm_provider}")
    print(f"  Model:        {config.llm_model}")
    print(f"  Embedding:    {config.embedding_provider} ({config.embedding_model})")
    print(f"  Database:     {config.db_path}")
    print(f"  Interface:    http://{args.host}:{args.port}")
    print(f"╚════════════════════════════════════════════╝\n")

    uvicorn.run(app, host=args.host, port=args.port)
