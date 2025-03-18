import numpy as np

class QuantumInspiredDifferentialEvolutionV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.F = 0.5
        self.CR = 0.9
        self.q_prob = 0.05  # Probability to use quantum-inspired move (adjusted from 0.1 to 0.05)
        self.scaling_layer = 2  # New scaling layer factor

    def quantum_move(self, individual, best, lb, ub):
        factor = np.random.uniform(0.5, 1.5) * self.scaling_layer
        new_position = best + factor * (individual - best)
        return np.clip(new_position, lb, ub)

    def adaptive_mutation(self):
        return np.random.uniform(0.4, 0.9)  # Broadened mutation range for better diversity

    def quantum_crossover(self, individual, best):
        q_prob_adaptive = np.clip(self.q_prob + np.random.uniform(-0.02, 0.02), 0.05, 0.1)  # Adaptive quantum probability
        return np.where(np.random.rand(self.dim) < q_prob_adaptive, best, individual)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        scores = np.full(self.population_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                self.F = self.adaptive_mutation()
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, lb, ub)

                trial = self.quantum_crossover(mutant, global_best_position)

                trial_score = func(trial)
                evaluations += 1
                if trial_score < scores[i]:
                    population[i] = trial
                    scores[i] = trial_score

                if np.random.rand() < self.q_prob:
                    population[i] = self.quantum_move(population[i], global_best_position, lb, ub)

                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = population[i].copy()

        return global_best_position, global_best_score