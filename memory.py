import sqlite3
import numpy as np
import pickle
import copy
import json
from datetime import datetime
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

from mind_core import Residue, Attractor


class PersistentRAM:
    """SQLite-backed persistent memory with attractor/residue storage."""

    def __init__(self, db_path: str, similarity_threshold: float = 0.75):
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.attractors: List[Attractor] = []
        self._init_db()
        self._load_from_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("PRAGMA journal_mode=WAL")
            c.execute('''CREATE TABLE IF NOT EXISTS attractors (
                id TEXT PRIMARY KEY,
                centroid BLOB,
                weight REAL,
                created_at TEXT,
                updated_at TEXT
            )''')
            c.execute('''CREATE TABLE IF NOT EXISTS residues (
                id TEXT PRIMARY KEY,
                layer_origin INTEGER,
                text TEXT,
                embedding BLOB,
                novelty REAL,
                stability REAL,
                epistemic_value REAL,
                attractor_weight REAL,
                r_score REAL,
                attractor_id TEXT,
                created_at TEXT
            )''')
            c.execute('''CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT,
                content TEXT,
                timestamp TEXT,
                emotional_valence REAL
            )''')
            c.execute('''CREATE TABLE IF NOT EXISTS personality_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )''')
            conn.commit()

    def _load_from_db(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT id, centroid, weight FROM attractors")
            rows = c.fetchall()
            for row in rows:
                a = Attractor(row[0])
                if row[1]:
                    a.centroid = pickle.loads(row[1])
                a.weight = row[2]
                c2 = conn.cursor()
                c2.execute("""SELECT id, layer_origin, text, embedding, novelty,
                              stability, epistemic_value, attractor_weight, r_score
                              FROM residues WHERE attractor_id=?""", (a.id,))
                for rrow in c2.fetchall():
                    r = Residue(rrow[0], rrow[1], rrow[2])
                    r.embedding = pickle.loads(rrow[3])
                    r.novelty = rrow[4]
                    r.stability = rrow[5]
                    r.epistemic_value = rrow[6]
                    r.attractor_weight = rrow[7]
                    r.R_score = rrow[8]
                    a.connected_residues.append(r)
                self.attractors.append(a)

    def save_attractor(self, a: Attractor):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            centroid_blob = pickle.dumps(a.centroid) if a.centroid is not None else None
            c.execute("""INSERT OR REPLACE INTO attractors
                         (id, centroid, weight, created_at, updated_at)
                         VALUES (?, ?, ?, ?, ?)""",
                      (a.id, centroid_blob, a.weight,
                       datetime.now().isoformat(), datetime.now().isoformat()))
            conn.commit()

    def save_residue(self, r: Residue, attractor_id: str):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            emb_blob = pickle.dumps(r.embedding)
            c.execute("""INSERT OR REPLACE INTO residues
                         (id, layer_origin, text, embedding, novelty, stability,
                          epistemic_value, attractor_weight, r_score, attractor_id, created_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                      (r.id, r.layer_origin, r.text, emb_blob, r.novelty, r.stability,
                       r.epistemic_value, r.attractor_weight, r.R_score,
                       attractor_id, datetime.now().isoformat()))
            conn.commit()

    def log_conversation(self, role: str, content: str, emotional_valence: float = 0.0):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""INSERT INTO conversations (role, content, timestamp, emotional_valence)
                         VALUES (?, ?, ?, ?)""",
                      (role, content, datetime.now().isoformat(), emotional_valence))
            conn.commit()

    def get_recent_context(self, n: int = 10):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""SELECT role, content FROM conversations
                         ORDER BY timestamp DESC LIMIT ?""", (n,))
            rows = c.fetchall()
            return list(reversed(rows))

    def save_personality(self, personality_dict: dict):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM personality_state")
            c.execute("INSERT INTO personality_state (key, value) VALUES (?, ?)",
                      ("personality", json.dumps(personality_dict)))
            conn.commit()

    def load_personality(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT value FROM personality_state WHERE key=?", ("personality",))
            row = c.fetchone()
            if row:
                return json.loads(row[0])
            return None

    def integrate_residues(self, residues: List[Residue], alpha=0.05,
                           lambda_entropy=0.1, delta_decay=0.01):
        total_weight = sum(a.weight for a in self.attractors) + 1e-6
        for r in residues:
            target = self.find_or_create_attractor(r)
            target.connected_residues.append(r)
            r.attractor_weight = target.weight
            target.amplify([r], alpha, 0.3, lambda_entropy, delta_decay, total_weight)
            target.update_centroid()
            self.save_residue(r, target.id)

    def find_or_create_attractor(self, residue: Residue) -> Attractor:
        for a in self.attractors:
            if a.centroid is not None:
                sim = cosine_similarity([a.centroid], [residue.embedding])[0][0]
                if sim > self.similarity_threshold:
                    return a
        attractor = Attractor(f"A_{residue.id}_{len(self.attractors)}")
        self.attractors.append(attractor)
        self.save_attractor(attractor)
        return attractor

    def propagate_influence(self, noise_scale: float = 0.005):
        for a in self.attractors:
            for r in a.connected_residues:
                drift = np.random.normal(0, noise_scale, r.embedding.shape)
                r.embedding += drift
                norm = np.linalg.norm(r.embedding)
                if norm > 0:
                    r.embedding = r.embedding / norm

    def prune(self, min_weight: float = 0.05, min_residues: int = 2):
        survivors = []
        removed_ids = []
        for a in self.attractors:
            if a.weight >= min_weight and len(a.connected_residues) >= min_residues:
                survivors.append(a)
            else:
                removed_ids.append(a.id)
        self.attractors = survivors
        if removed_ids:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                for rid in removed_ids:
                    c.execute("DELETE FROM residues WHERE attractor_id=?", (rid,))
                    c.execute("DELETE FROM attractors WHERE id=?", (rid,))
                conn.commit()

    def clone(self):
        return copy.deepcopy(self)

    def get_stats(self):
        return {
            "attractors": len(self.attractors),
            "total_residues": sum(len(a.connected_residues) for a in self.attractors),
            "avg_weight": float(np.mean([a.weight for a in self.attractors])) if self.attractors else 0.0,
        }
