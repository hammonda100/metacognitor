import os
import sys
import random
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MindConfig
from llm_client import LLMClient
from embeddings import EmbeddingClient
from memory import PersistentRAM
from personality import Personality
from mind_core import Residue, MindCycle


class TerminalMind:
    def __init__(self):
        self.config = MindConfig.from_env()
        self.llm_client = LLMClient(self.config)
        self.embedder = EmbeddingClient(self.config)
        self.ram = PersistentRAM(self.config.db_path, self.config.similarity_threshold)

        personality_data = self.ram.load_personality()
        self.personality = Personality.from_dict(personality_data) if personality_data else Personality()

        self.system_state = {
            'cognition_patterns': [],
            'mind_agents': [],
            'life_signals': 0.0,
            'residue_growth': 0
        }

    async def initialize(self):
        print("\n[Initializing Mind Cycle...]")
        energy_field = [random.random() for _ in range(10)]
        self.system_state = await MindCycle(
            0, self.system_state, self.ram, energy_field,
            self.llm_client, self.embedder, max_depth=2, personality=self.personality
        )
        print("[Mind initialized with {} attractors, {} residues]\n".format(
            self.ram.get_stats()['attractors'],
            self.ram.get_stats()['total_residues']
        ))

    def print_stats(self):
        stats = self.ram.get_stats()
        p = self.personality
        print(f"\n  ┌─ Memory ─────────────────────┐")
        print(f"  │ Attractors:     {stats['attractors']:>10} │")
        print(f"  │ Residues:       {stats['total_residues']:>10} │")
        print(f"  │ Avg Weight:      {stats['avg_weight']:>9.3f} │")
        print(f"  ├─ Personality ──────────────────┤")
        print(f"  │ Interactions:    {p.interaction_count:>10} │")
        print(f"  │ Curiosity:       {p.traits['curiosity']:>9.2f} │")
        print(f"  │ Openness:        {p.traits['openness']:>9.2f} │")
        print(f"  │ Empathy:         {p.traits['empathy']:>9.2f} │")
        print(f"  │ Playfulness:     {p.traits['playfulness']:>9.2f} │")
        print(f"  └────────────────────────────────┘\n")

    async def respond(self, user_input: str):
        features = self.personality.analyze_input(user_input)
        self.ram.log_conversation("user", user_input, features['sentiment'])

        recent = self.ram.get_recent_context(n=10)
        context_lines = []
        for role, content in recent:
            prefix = "User" if role == "user" else "Mind"
            context_lines.append(f"{prefix}: {content}")
        context = "\n".join(context_lines[-5:])

        system_prompt = self.personality.generate_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Recent context:\n{context}\n\nUser says: {user_input}"}
        ]

        response = await self.llm_client.chat(messages, max_tokens=self.config.llm_max_tokens)
        self.ram.log_conversation("assistant", response, 0.0)

        # Residue from response
        r = Residue(id=f"user_res_{random.randint(0,100000)}", layer_origin=0, text=response)
        r.embedding = await self.embedder.aembed(response)
        r.compute_R_score()
        self.ram.integrate_residues([r])
        self.system_state['cognition_patterns'].append(response)

        # Mind cycle
        energy_field = [random.random() for _ in range(10)]
        if features['sentiment'] > 0.3:
            energy_field = [min(1.0, e + 0.1) for e in energy_field]
        elif features['sentiment'] < -0.3:
            energy_field = [max(0.0, e - 0.1) for e in energy_field]

        self.system_state = await MindCycle(
            0, self.system_state, self.ram, energy_field,
            self.llm_client, self.embedder, max_depth=2, personality=self.personality
        )

        self.personality.evolve(user_input, response, self.system_state['residue_growth'])
        self.ram.save_personality(self.personality.to_dict())

        return response

    async def run(self):
        await self.initialize()
        print("=== Continuous Learning Mind v2.0 ===")
        print("Commands: /stats, /personality, /memory, /exit\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
                print("\nSession ended. Mind persists in memory.")
                break
            elif user_input == "/stats":
                self.print_stats()
                continue
            elif user_input == "/personality":
                print("\n" + json.dumps(self.personality.to_dict(), indent=2) + "\n")
                continue
            elif user_input == "/memory":
                stats = self.ram.get_stats()
                print(f"\nMemory: {stats['attractors']} attractors, {stats['total_residues']} residues\n")
                continue

            try:
                response = await self.respond(user_input)
                print(f"Mind: {response}\n")
            except Exception as e:
                print(f"[Error: {e}]\n")


if __name__ == "__main__":
    import json
    mind = TerminalMind()
    asyncio.run(mind.run())
