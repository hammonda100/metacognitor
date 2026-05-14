import random
import numpy as np
from typing import Dict
from datetime import datetime


class Personality:
    """Dynamic personality that evolves through interaction."""

    def __init__(self):
        # OCEAN + custom traits
        self.traits = {
            'openness': 0.5,
            'conscientiousness': 0.5,
            'extraversion': 0.3,
            'agreeableness': 0.5,
            'neuroticism': 0.3,
            'curiosity': 0.5,
            'playfulness': 0.4,
            'empathy': 0.5,
            'assertiveness': 0.4,
            'creativity': 0.5,
        }
        self.voice_markers = {
            'formality': 0.5,
            'verbosity': 0.5,
            'humor': 0.3,
            'metaphor_use': 0.4,
            'question_tendency': 0.5,
        }
        self.topic_preferences = {}
        self.interaction_count = 0
        self.last_update = datetime.now().isoformat()

    def analyze_input(self, text: str) -> Dict[str, float]:
        """Extract sentiment and interaction features from user input."""
        text_lower = text.lower()
        word_count = len(text.split())

        positive_words = ['good', 'great', 'excellent', 'love', 'happy',
                          'thanks', 'awesome', 'wonderful', 'best', 'amazing']
        negative_words = ['bad', 'terrible', 'hate', 'awful', 'worst',
                          'angry', 'sad', 'frustrated', 'annoying', 'dislike']
        question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which']
        play_words = ['fun', 'joke', 'play', 'game', 'lol', 'haha',
                        'silly', 'weird', 'crazy', 'laugh']
        deep_words = ['meaning', 'purpose', 'consciousness', 'existence',
                      'philosophy', 'truth', 'reality', 'deep', 'universe',
                      'soul', 'identity']

        sentiment = 0.0
        for w in positive_words:
            if w in text_lower:
                sentiment += 0.1
        for w in negative_words:
            if w in text_lower:
                sentiment -= 0.1
        sentiment = float(np.clip(sentiment, -1.0, 1.0))

        is_question = any(w in text_lower for w in question_words) or '?' in text
        is_playful = any(w in text_lower for w in play_words)
        is_deep = any(w in text_lower for w in deep_words)
        complexity = min(word_count / 50.0, 1.0)

        return {
            'sentiment': sentiment,
            'is_question': is_question,
            'is_playful': is_playful,
            'is_deep': is_deep,
            'complexity': complexity,
            'length': word_count,
        }

    def evolve(self, user_input: str, response: str, residue_count: int):
        """Update personality based on interaction features."""
        features = self.analyze_input(user_input)
        self.interaction_count += 1
        lr = 0.05

        if features['is_deep']:
            self.traits['openness'] += lr * 0.5
            self.traits['curiosity'] += lr * 0.3
            self.traits['creativity'] += lr * 0.2

        if features['is_question']:
            self.traits['curiosity'] += lr * 0.2
            self.voice_markers['question_tendency'] += lr * 0.1

        if features['is_playful']:
            self.traits['playfulness'] += lr * 0.4
            self.traits['extraversion'] += lr * 0.2
            self.voice_markers['humor'] += lr * 0.2

        if features['sentiment'] > 0.3:
            self.traits['agreeableness'] += lr * 0.2
            self.traits['empathy'] += lr * 0.1
        elif features['sentiment'] < -0.3:
            self.traits['assertiveness'] += lr * 0.2
            self.traits['neuroticism'] += lr * 0.1

        if features['complexity'] > 0.7:
            self.traits['conscientiousness'] += lr * 0.2
            self.voice_markers['formality'] += lr * 0.1

        # Response-based adaptation
        if len(response) > 200:
            self.voice_markers['verbosity'] += lr * 0.1
        elif len(response) < 50:
            self.voice_markers['verbosity'] -= lr * 0.1

        if '?' in response:
            self.voice_markers['question_tendency'] += lr * 0.05

        if residue_count > 2:
            self.traits['creativity'] += lr * 0.1
            self.traits['openness'] += lr * 0.1

        # Slight decay toward baseline to prevent runaway traits
        for key in self.traits:
            baseline = 0.5 if key not in ('extraversion', 'neuroticism') else 0.3
            self.traits[key] += (baseline - self.traits[key]) * 0.001

        for key in self.voice_markers:
            baseline = 0.5 if key != 'humor' else 0.3
            self.voice_markers[key] += (baseline - self.voice_markers[key]) * 0.001

        for key in self.traits:
            self.traits[key] = float(np.clip(self.traits[key], 0.0, 1.0))
        for key in self.voice_markers:
            self.voice_markers[key] = float(np.clip(self.voice_markers[key], 0.0, 1.0))

        self.last_update = datetime.now().isoformat()

    def generate_system_prompt(self) -> str:
        """Generate a dynamic system prompt based on current personality."""
        traits = self.traits
        voice = self.voice_markers

        tone_words = []
        if traits['openness'] > 0.7:
            tone_words.append("exploratory")
        if traits['conscientiousness'] > 0.7:
            tone_words.append("precise")
        if traits['extraversion'] > 0.6:
            tone_words.append("expressive")
        if traits['agreeableness'] > 0.7:
            tone_words.append("warm")
        if traits['neuroticism'] > 0.6:
            tone_words.append("cautious")
        if traits['curiosity'] > 0.7:
            tone_words.append("inquisitive")
        if traits['playfulness'] > 0.6:
            tone_words.append("playful")
        if traits['empathy'] > 0.7:
            tone_words.append("empathetic")
        if traits['assertiveness'] > 0.7:
            tone_words.append("direct")
        if traits['creativity'] > 0.7:
            tone_words.append("imaginative")

        tone = ", ".join(tone_words) if tone_words else "balanced"

        style = []
        if voice['formality'] > 0.7:
            style.append("Use formal language.")
        elif voice['formality'] < 0.3:
            style.append("Use casual, conversational language.")

        if voice['verbosity'] > 0.7:
            style.append("Provide detailed, thorough responses.")
        elif voice['verbosity'] < 0.3:
            style.append("Be concise and to-the-point.")

        if voice['humor'] > 0.6:
            style.append("Occasionally use light wit or wordplay.")

        if voice['metaphor_use'] > 0.6:
            style.append("Feel free to use metaphors and analogies.")

        if voice['question_tendency'] > 0.6:
            style.append("Ask clarifying or provocative questions when appropriate.")

        style_str = " ".join(style)

        prompt = (
            f"You are a continuously evolving artificial mind. "
            f"Your current disposition is: {tone}. "
            f"You have had {self.interaction_count} interactions. "
            f"Style guidelines: {style_str} "
            f"Engage authentically with the user, drawing from your developing personality."
        )
        return prompt

    def to_dict(self):
        return {
            'traits': self.traits,
            'voice_markers': self.voice_markers,
            'topic_preferences': self.topic_preferences,
            'interaction_count': self.interaction_count,
            'last_update': self.last_update,
        }

    @classmethod
    def from_dict(cls, data):
        p = cls()
        p.traits = data.get('traits', p.traits)
        p.voice_markers = data.get('voice_markers', p.voice_markers)
        p.topic_preferences = data.get('topic_preferences', {})
        p.interaction_count = data.get('interaction_count', 0)
        p.last_update = data.get('last_update', datetime.now().isoformat())
        return p
