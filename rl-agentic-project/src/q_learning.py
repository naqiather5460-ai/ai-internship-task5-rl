"""Tabular Q-Learning agent (off-policy TD control).

Update rule:
    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
"""
import numpy as np


class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999, seed=0):
        self.q = np.zeros((n_states, n_actions))
        self.n_actions = n_actions
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, epsilon_min, epsilon_decay
        self.rng = np.random.default_rng(seed)

    def act(self, state, greedy=False):
        """Epsilon-greedy policy: explore with prob. epsilon, else exploit."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        best = np.flatnonzero(self.q[state] == self.q[state].max())
        return int(self.rng.choice(best))  # random tie-break

    def update(self, s, a, r, s_next, terminated):
        # Bootstrap only if the episode did not truly end (terminated, not truncated)
        target = r + (0.0 if terminated else self.gamma * self.q[s_next].max())
        self.q[s, a] += self.alpha * (target - self.q[s, a])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
