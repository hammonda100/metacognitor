import os
import json
import random
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from config import MindConfig
from llm_client import LLMClient
from embeddings import EmbeddingClient
from memory import PersistentRAM
from personality import Personality
from mind_core import Residue, MindCycle

app = FastAPI(title="Continuous Learning Mind")
config = MindConfig.from_env()
llm_client = LLMClient(config)
embedder = EmbeddingClient(config)
ram = PersistentRAM(config.db_path, config.similarity_threshold)

# Load or initialize personality
personality_data = ram.load_personality()
personality = Personality.from_dict(personality_data) if personality_data else Personality()

system_state = {
    'cognition_patterns': [],
    'mind_agents': [],
    'life_signals': 0.0,
    'residue_growth': 0
}

templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def startup():
    global system_state
    energy_field = [random.random() for _ in range(10)]
    system_state = await MindCycle(
        0, system_state, ram, energy_field,
        llm_client, embedder, max_depth=2, personality=personality
    )

@app.get("/", response_class=HTMLResponse)
async def get_chat(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/stats")
async def get_stats():
    return {
        "memory": ram.get_stats(),
        "personality": personality.to_dict(),
        "system_state": {
            k: v for k, v in system_state.items()
            if k != 'cognition_patterns' or len(v) < 100
        }
    }

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            user_input = msg.get("text", "")

            if not user_input:
                continue

            # Analyze and log
            features = personality.analyze_input(user_input)
            ram.log_conversation("user", user_input, features['sentiment'])

            # Build context
            recent = ram.get_recent_context(n=10)
            context_lines = []
            for role, content in recent:
                prefix = "User" if role == "user" else "Mind"
                context_lines.append(f"{prefix}: {content}")
            context = "\n".join(context_lines[-5:])

            # Dynamic system prompt from evolving personality
            system_prompt = personality.generate_system_prompt()

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Recent context:\n{context}\n\nUser says: {user_input}"}
            ]

            response = await llm_client.chat(
                messages, max_tokens=config.llm_max_tokens
            )

            # Log response
            ram.log_conversation("assistant", response, 0.0)

            # Create residue from interaction
            r = Residue(
                id=f"user_res_{random.randint(0, 100000)}",
                layer_origin=0, text=response
            )
            r.embedding = await embedder.aembed(response)
            r.compute_R_score()
            ram.integrate_residues([r])
            system_state['cognition_patterns'].append(response)

            # Run mind cycle with sentiment-influenced energy
            energy_field = [random.random() for _ in range(10)]
            if features['sentiment'] > 0.3:
                energy_field = [min(1.0, e + 0.1) for e in energy_field]
            elif features['sentiment'] < -0.3:
                energy_field = [max(0.0, e - 0.1) for e in energy_field]

            system_state = await MindCycle(
                0, system_state, ram, energy_field,
                llm_client, embedder, max_depth=2, personality=personality
            )

            # Evolve personality based on this interaction
            personality.evolve(user_input, response, system_state['residue_growth'])
            ram.save_personality(personality.to_dict())

            # Send response with metadata
            await websocket.send_json({
                "response": response,
                "personality_snapshot": personality.to_dict(),
                "memory_stats": ram.get_stats(),
                "residue_growth": system_state['residue_growth'],
                "life_signals": system_state['life_signals'],
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
        finally:
            await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)
