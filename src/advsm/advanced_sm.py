from datetime import datetime, timedelta
from supermemo2 import first_review, review
from advsm.memory_state_machine import MemoryStateMachine


class AdvancedSuperMemo:
    """
    SM-2 turbinado com Cadeias de Markov + tempo de resposta + tipo de assunto.
    """

    def __init__(self):
        self.machine = MemoryStateMachine()

    # ---------------------- heurísticas de classificação --------------- #
    @staticmethod
    def _classify_state(response_time: float, subject: str) -> str:
        """
        response_time  – segundos até responder
        subject        – 'easy' | 'medium' | 'hard'
        """
        if subject == "hard":
            return "Weak" if response_time > 10 else "Medium"
        if subject == "medium":
            return "Good" if response_time <= 6 else "Medium"
        # 'easy'
        return "Excellent" if response_time <= 5 else "Good"

    # ------------------------- rotina de revisão ----------------------- #
    def markov_review(
        self,
        quality: int,
        response_time_sec: float,
        subject_type: str,
        review_info: dict | None = None,
    ) -> dict:
        """
        Retorna um dicionário:
            • valores SM-2 habituales
            • 'memory_state': estado Markov observado
            • 'expected_forgetting_prob': P(Forgotten | memory_state)
        """
        observed_state = self._classify_state(response_time_sec, subject_type)

        # 1) Chamada SM-2 pura
        if review_info is None:
            sm = first_review(quality)
            previous_state = observed_state                      # primeira ocorrência
        else:
            sm = review(quality, **review_info)
            previous_state = self._classify_state(
                response_time_sec, subject_type
            )  # heurística simples

        # 2) Aprendizagem da cadeia de Markov
        self.machine.update(previous_state, observed_state)

        # 3) Previsão Markov & ajuste de intervalo
        predicted_state = self.machine.next_state(observed_state)
        adjust = {
            "Excellent": 1.3,
            "Good": 1.1,
            "Medium": 0.9,
            "Weak": 0.7,
            "Forgotten": 0.5,
        }[predicted_state]

        sm["interval"] = max(1, int(sm["interval"] * adjust))
        sm["review_datetime"] = datetime.utcnow() + timedelta(days=sm["interval"])

        # 4) Métricas extras
        sm["memory_state"] = observed_state
        sm["expected_forgetting_prob"] = self.machine.forgetting_probability(
            observed_state
        )

        return sm
