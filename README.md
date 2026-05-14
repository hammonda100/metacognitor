<div align="center">

# 🧠 Continuous Learning Mind

**A self-evolving cognitive architecture with persistent memory, dynamic personality, and universal LLM support.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Every conversation shapes the mind. Every interaction evolves the character.*

</div>

---

## ✨ What This Is

The **Continuous Learning Mind** is a recursive cognitive system that doesn't just respond — it **learns**, **remembers**, and **grows a personality** based on your interactions. It combines:

- 🧩 **Semantic memory** (attractor/residue clustering with embeddings)
- 🎭 **Dynamic personality evolution** (OCEAN + custom traits that shift over time)
- 🔌 **Universal LLM support** (OpenAI, Anthropic, Ollama, or any OpenAI-compatible API)
- 💾 **Persistent SQLite memory** (survives restarts)
- 🌐 **Real-time web chat** (WebSocket-powered live interface)
- 🖥️ **Terminal mode** (for headless / SSH use)

The mind runs recursive **MindCycles** that generate novel conceptual residues, cluster them into semantic attractors, and use the resulting cognitive landscape to inform every future response.

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/continuous-learning-mind.git
cd continuous-learning-mind

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure your LLM provider
cp .env.example .env
# Edit .env with your API keys and preferred model

# Launch the web interface
python main.py
# → Open http://localhost:8000 in your browser
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         USER INPUT                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  SENTIMENT ANALYSIS  →  CONVERSATION LOGGED TO SQLITE       │
│  (positive/negative/neutral feature extraction)             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  CONTEXT ASSEMBLY                                           │
│  • Last 10 conversation turns                               │
│  • Dynamic system prompt from evolving personality            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  LLM RESPONSE GENERATION                                    │
│  (OpenAI / Anthropic / Ollama / OpenAI-compatible)          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  RESPONSE → RESIDUE                                         │
│  • Embedded via sentence-transformers or OpenAI embeddings  │
│  • Scored via R = (n×s×e×a) + γ(n+s+e+a)                    │
│  • Integrated into attractor network                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  RECURSIVE MIND CYCLE (depth ≤ max_depth)                   │
│  • BHO generates new residues at high-energy points         │
│  • CIE integrates into cognition patterns                   │
│  • LDM detects life signals                                 │
│  • LGM spawns mind agents from high-R residues              │
│  • RM decides whether to recurse deeper                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  PERSONALITY EVOLUTION                                      │
│  • Traits shift based on input sentiment, complexity, depth   │
│  • Voice markers adapt to response patterns                   │
│  • Decay toward baseline prevents runaway drift             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  PERSISTENCE                                                │
│  • Attractors, residues, conversations → SQLite            │
│  • Personality state → SQLite                               │
│  • WAL mode for safe concurrent access                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎭 Dynamic Personality

The mind doesn't have a fixed persona. It develops one through **10 evolving traits** and **5 voice markers**:

### Traits (OCEAN + Custom)
| Trait | Influenced By | Effect |
|-------|---------------|--------|
| **Openness** | Deep/philosophical inputs | More exploratory, metaphorical |
| **Conscientiousness** | Complex, detailed inputs | More precise, structured |
| **Extraversion** | Playful, social language | More expressive, energetic |
| **Agreeableness** | Positive sentiment | Warmer, more supportive |
| **Neuroticism** | Negative sentiment | More cautious, direct |
| **Curiosity** | Questions, exploration | More inquisitive responses |
| **Playfulness** | Jokes, games, humor | Lighter, wittier tone |
| **Empathy** | Emotional sharing | More emotionally attuned |
| **Assertiveness** | Criticism, conflict | More direct, firm |
| **Creativity** | High residue growth | More imaginative, novel |

### Voice Markers
- **Formality** — shifts between casual and formal register
- **Verbosity** — adapts response length to user preference
- **Humor** — gradually incorporates wit when playful inputs detected
- **Metaphor use** — increases with openness and creativity
- **Question tendency** — rises when user asks many questions

**Every LLM call uses a dynamically generated system prompt** based on the current personality state, so the mind's voice genuinely changes over time.

---

## 🔌 LLM Providers

Switch providers by setting environment variables (or editing `.env`):

### OpenAI
```bash
MIND_LLM_PROVIDER=openai
MIND_LLM_MODEL=gpt-4
MIND_LLM_API_KEY=sk-...
```

### Anthropic Claude
```bash
MIND_LLM_PROVIDER=anthropic
MIND_LLM_MODEL=claude-3-sonnet-20240229
MIND_LLM_API_KEY=sk-ant-...
```

### Local Ollama
```bash
MIND_LLM_PROVIDER=ollama
MIND_LLM_MODEL=llama3
MIND_LLM_BASE_URL=http://localhost:11434
```

### Any OpenAI-compatible API (vLLM, LM Studio, etc.)
```bash
MIND_LLM_PROVIDER=openai_compatible
MIND_LLM_BASE_URL=http://localhost:8000/v1
MIND_LLM_API_KEY=your-key  # if required
```

---

## 📁 Project Structure

```
continuous-learning-mind/
├── config.py              # Centralized configuration (dataclass + env vars)
├── llm_client.py          # Unified LLM provider abstraction
├── embeddings.py           # Embedding provider (local / OpenAI)
├── mind_core.py            # Core cognitive architecture
│   ├── Residue             # Semantic concept with embedding + scoring
│   ├── Attractor           # Cluster that binds related residues
│   ├── BHO                 # Break/Heal Operator (LLM-driven residue gen)
│   ├── CIE                 # Cognitive Integration Engine
│   ├── LDM                 # Life Detection Module
│   ├── LGM                 # Life Generation Module
│   ├── RM                  # Recursion Modulator
│   └── MindCycle           # Recursive cognitive loop
├── memory.py               # PersistentRAM (SQLite-backed attractors + residues)
├── personality.py          # Dynamic personality evolution engine
├── web_app.py              # FastAPI + WebSocket server
├── cli.py                  # Terminal interactive mode
├── main.py                 # Entry point for web server
├── templates/
│   └── index.html          # Real-time chat UI with live personality viz
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🖥️ Usage Modes

### Web Chat (Recommended)
```bash
python main.py
# → http://localhost:8000
```

Features:
- WebSocket-based real-time messaging
- Live sidebar: memory stats, personality trait bars, energy field visualization
- Dark futuristic UI with animated status indicators

### Terminal Mode
```bash
python cli.py
```

Commands:
| Command | Description |
|---------|-------------|
| `/stats` | Show memory and personality statistics |
| `/personality` | Dump full personality JSON |
| `/memory` | Show attractor/residue counts |
| `/exit` | Quit (state persists in SQLite) |

---

## ⚙️ Configuration

All settings are controlled via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MIND_LLM_PROVIDER` | `openai` | LLM backend: `openai`, `anthropic`, `ollama`, `openai_compatible` |
| `MIND_LLM_MODEL` | `gpt-4` | Model name |
| `MIND_LLM_API_KEY` | — | API key (or `OPENAI_API_KEY`) |
| `MIND_LLM_BASE_URL` | — | Custom base URL for compatible APIs |
| `MIND_LLM_TEMPERATURE` | `0.7` | Sampling temperature |
| `MIND_LLM_MAX_TOKENS` | `150` | Max tokens per response |
| `MIND_EMBEDDING_PROVIDER` | `local` | `local` (sentence-transformers) or `openai` |
| `MIND_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model name |
| `MIND_DB_PATH` | `./mind_memory.db` | SQLite database path |
| `MIND_SIMILARITY_THRESHOLD` | `0.75` | Cosine similarity threshold for attractor binding |
| `MIND_MAX_DEPTH` | `3` | Maximum MindCycle recursion depth |

---

## 🔬 How It Works (Detailed)

### 1. Residue Scoring (R-Score)
Each residue is scored by:

```
R = (novelty × stability × epistemic_value × attractor_weight)
    + γ × (novelty + stability + epistemic_value + attractor_weight)
```

Where `γ = 0.1`. This ensures residues with balanced high traits score highest — not just novel or just stable, but meaningfully integrated.

### 2. Attractor Dynamics
Attractors cluster residues by cosine similarity (> threshold). Their weight evolves:

```
weight += α × R_score          # amplification from new residues
weight -= λ × (p - p_max)      # entropy penalty if too dominant
weight *= (1 - δ)              # natural decay
```

Weak attractors (weight < 0.05 or < 2 residues) are pruned from memory.

### 3. Personality Evolution
Traits shift via reinforcement learning with decay:

```
trait += learning_rate × influence
# + slight decay toward baseline to prevent runaway
```

For example, asking philosophical questions increases `openness` and `curiosity`, while telling jokes increases `playfulness` and `extraversion`. Over hundreds of interactions, the mind develops a genuinely distinct character.

### 4. Persistence
Everything survives restarts:
- **Attractors** with centroids and weights
- **Residues** with embeddings, scores, and metadata
- **Conversation history** (last N turns loaded for context)
- **Personality state** (full trait and voice marker snapshot)

---

## 🧪 Extending the Mind

### Add a New LLM Provider
Implement `BaseLLMProvider` in `llm_client.py`:

```python
class MyProvider(BaseLLMProvider):
    async def chat(self, messages, **kwargs) -> str:
        # Your implementation
        pass

    async def generate(self, prompt, num_residues=3, **kwargs) -> List[str]:
        # Your implementation
        pass
```

Then register it in `LLMClient.__init__`.

### Add New Personality Traits
1. Add to `Personality.traits` dictionary
2. Add evolution logic in `Personality.evolve()`
3. Add voice marker effects in `Personality.generate_system_prompt()`

### Custom Embedding Models
Implement `BaseEmbeddingProvider` in `embeddings.py` and register in `EmbeddingClient`.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with curiosity. Evolved through conversation.**

</div>
