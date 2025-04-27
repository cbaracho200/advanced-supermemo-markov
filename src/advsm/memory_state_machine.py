import numpy as np
from collections import defaultdict

class MemoryStateMachine:
    """
    Cadeia de Markov discreta que aprende on-line.
    Estados fixos:
        0 – Excellent
        1 – Good
        2 – Medium
        3 – Weak
        4 – Forgotten
    """
    states = ["Excellent", "Good", "Medium", "Weak", "Forgotten"]

    def __init__(self, alpha: float = 1.0):
        # Conte os fluxos entre estados (Dirichlet(α) como prior)
        self.counts = np.full((5, 5), alpha, dtype=float)

    # ------------------------------------------------------------------ #
    #  Transições & Probabilidades                                       #
    # ------------------------------------------------------------------ #
    def _probs(self, row_idx: int) -> np.ndarray:
        row = self.counts[row_idx]
        return row / row.sum()

    def next_state(self, current_state: str) -> str:
        idx = self.states.index(current_state)
        next_idx = np.random.choice(5, p=self._probs(idx))
        return self.states[next_idx]

    def update(self, current_state: str, observed_state: str) -> None:
        i, j = self.states.index(current_state), self.states.index(observed_state)
        self.counts[i, j] += 1                      # aprendizagem Bayesiana

    # ------------------------------------------------------------------ #
    #  Métricas                                                          #
    # ------------------------------------------------------------------ #
    def forgetting_probability(self, state: str) -> float:
        """Probabilidade de virar Forgotten no próximo passo."""
        idx = self.states.index(state)
        return self._probs(idx)[4]                 # coluna 4 = Forgotten

    def transition_matrix(self) -> np.ndarray:
        """Matriz 5×5 normalizada (cópia)."""
        return np.stack([self._probs(i) for i in range(5)], axis=0).copy()
