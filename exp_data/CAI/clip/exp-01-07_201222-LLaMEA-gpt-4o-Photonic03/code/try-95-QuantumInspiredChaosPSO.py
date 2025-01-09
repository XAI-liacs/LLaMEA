import numpy as np

class QuantumInspiredChaosPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.c1 = 1.5   # Cognitive component
        self.c2 = 2.0   # Social component
        self.w = 0.5    # Inertia weight
        self.chaotic_factor = 0.7  # Initial chaotic factor
        self.beta = 0.5  # Quantum factor

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = pop.copy()
        personal_best_scores = np.array([func(ind) for ind in personal_best])

        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.population_size

        z = np.random.random(self.population_size)  # Chaotic variable

        while evaluations < self.budget:
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            z = self.chaotic_factor * z * (1 - z)  # Logistic map for chaotic sequence

            velocities = self.w * velocities + self.c1 * r1 * (personal_best - pop) + self.c2 * r2 * (global_best - pop)
            pop = pop + velocities + z[:, np.newaxis] * (np.random.uniform(lb, ub, (self.population_size, self.dim)) - pop)
            pop = np.clip(pop, lb, ub)

            scores = np.array([func(ind) for ind in pop])
            evaluations += self.population_size

            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best[i] = pop[i]

            if np.min(scores) < global_best_score:
                global_best_score = np.min(scores)
                global_best = pop[np.argmin(scores)]

            # Quantum-inspired update
            for i in range(self.population_size):
                quantum_jump = self.beta * (np.random.uniform(lb, ub, self.dim) - pop[i])
                quantum_candidate = pop[i] + quantum_jump
                quantum_candidate = np.clip(quantum_candidate, lb, ub)
                quantum_score = func(quantum_candidate)
                evaluations += 1
                if quantum_score < scores[i]:
                    scores[i] = quantum_score
                    pop[i] = quantum_candidate

        return global_best, global_best_score