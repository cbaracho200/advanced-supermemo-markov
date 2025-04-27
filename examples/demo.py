from advsm import AdvancedSuperMemo

sm = AdvancedSuperMemo()

# primeira revisão (rápida, assunto fácil)
r = sm.markov_review(quality=5, response_time_sec=3, subject_type="easy")
print("1ª revisão:", r)

# segunda revisão (assunto difícil, demorou)
r2 = sm.markov_review(
    quality=3, response_time_sec=14, subject_type="hard", review_info=r
)
print("2ª revisão:", r2)

# olhe a matriz aprendida
print("\nMatriz de transição atual:")
print(sm.machine.transition_matrix())
