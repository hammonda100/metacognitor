import copy
import random
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity


class Residue:
    """A conceptual residue with semantic embedding and dynamic scoring."""

    def __init__(self, id: str, layer_origin: int, text: str, embedding: np.ndarray = None):
        self.id = id
        self.layer_origin = layer_origin
        self.text = text
        self.embedding = embedding
        self.novelty = random.random()
        self.stability = random.random()
        self.epistemic_value = random.random()
        self.attractor_weight = 0.1  # FIX: Baseline instead of 0.0
        self.R_score = 0.0

    def compute_R_score(self, gamma: float = 0.1):
        mult = (max(self.novelty, 0.01) *
                max(self.stability, 0.01) *
                max(self.epistemic_value, 0.01) *
                max(self.attractor_weight, 0.01))
        add = gamma * (self.novelty + self.stability + self.epistemic_value + self.attractor_weight)
        self.R_score = mult + add
        return self.R_score


class Attractor:
    """Semantic cluster that binds related residues."""

    def __init__(self, id: str):
        self.id = id
        self.connected_residues: List[Residue] = []
        self.weight = 0.1
        self.centroid = None

    def update_centroid(self):
        if self.connected_residues:
            self.centroid = np.mean([r.embedding for r in self.connected_residues], axis=0)
            # FIX: Normalize instead of destructive [0,1] clipping
            norm = np.linalg.norm(self.centroid)
            if norm > 0:
                self.centroid = self.centroid / norm

    def amplify(self, new_residues: List[Residue], alpha: float = 0.05,
                p_max: float = 0.3, lambda_entropy: float = 0.1,
                delta_decay: float = 0.01, total_weight: float = 1.0):
        for r in new_residues:
            self.weight += alpha * r.R_score
        p = self.weight / max(total_weight, 1.0)
        if p > p_max:
            self.weight -= lambda_entropy * (p - p_max)
        self.weight *= (1 - delta_decay)
        self.update_centroid()


class BHO:
    """Break/Heal Operator - generates new residues via LLM."""

    @staticmethod
    async def apply(llm_client, point: int, layer: int):
        prompt = (f"Layer_{layer}_point_{point}_mind_concept: "
                  f"Generate 3 novel, concise conceptual residues (one per line).")
        concepts = await llm_client.generate(prompt, num_residues=3)
        residues = []
        for i, text in enumerate(concepts):
            r = Residue(id=f"R_{layer}_{point}_{i}", layer_origin=layer, text=text)
            r.compute_R_score()
            residues.append(r)
        return residues


class CIE:
    """Cognitive Integration Engine."""

    @staticmethod
    def integrate(residues: List[Residue], system_state: Dict[str, Any]):
        for r in residues:
            system_state['cognition_patterns'].append(r.text)
        # IMPROVEMENT: Prevent unbounded growth
        system_state['cognition_patterns'] = system_state['cognition_patterns'][-100:]
        return system_state


class LDM:
    """Life Detection Module."""

    @staticmethod
    def detect(residues: List[Residue], system_state: Dict[str, Any]):
        for r in residues:
            system_state['life_signals'] += r.R_score * 0.1
        return system_state


class LGM:
    """Life Generation Module - spawns mind agents from high-value residues."""

    @staticmethod
    def generate(residues: List[Residue], system_state: Dict[str, Any]):
        for r in residues:
            if r.R_score > 0.2:
                agent_id = f"agent_{r.id}"
                if agent_id not in system_state['mind_agents']:
                    system_state['mind_agents'].append(agent_id)
        # IMPROVEMENT: Cap agent list
        system_state['mind_agents'] = system_state['mind_agents'][-50:]
        return system_state


class RM:
    """Recursion Modulator."""

    @staticmethod
    def should_recurse(layer: int, ram_snapshot, max_depth: int):
        if layer >= max_depth:
            return False
        return any(a.weight > 0.1 and len(a.connected_residues) > 1
                   for a in ram_snapshot.attractors)


async def MindCycle(layer: int, system_state: Dict[str, Any], ram,
                    energy_field: List[float], llm_client, embedder,
                    max_depth: int = 5, personality=None):
    """Recursive cognitive cycle with personality-aware energy interpretation."""
    alpha = 0.03 + 0.02 * random.random()
    lambda_entropy = 0.05 + 0.05 * random.random()
    residue_threshold = 0.05 + 0.05 * random.random()

    # Personality influences energy field interpretation
    if personality:
        curiosity_boost = personality.traits.get('curiosity', 0.5)
        threshold_modifier = (curiosity_boost - 0.5) * 0.1
        residue_threshold += threshold_modifier

    insertion_points = [i for i, e in enumerate(energy_field)
                        if e > 0.5 + random.uniform(-0.1, 0.1)]
    high_value_residues = []

    for point in insertion_points:
        residues = await BHO.apply(llm_client, point, layer)
        for r in residues:
            # Async embedding to avoid blocking the event loop
            r.embedding = await embedder.aembed(r.text)
            r.compute_R_score()
            if r.R_score > residue_threshold:
                high_value_residues.append(r)

    ram.integrate_residues(high_value_residues, alpha, lambda_entropy)
    ram.propagate_influence()
    system_state = CIE.integrate(high_value_residues, system_state)
    system_state = LDM.detect(high_value_residues, system_state)
    system_state = LGM.generate(high_value_residues, system_state)

    ram_snapshot = ram.clone()
    if RM.should_recurse(layer, ram_snapshot, max_depth):
        system_state = await MindCycle(layer + 1, system_state, ram,
                                       energy_field, llm_client, embedder,
                                       max_depth, personality)

    ram.prune()
    system_state['residue_growth'] = len(high_value_residues)
    return system_state
