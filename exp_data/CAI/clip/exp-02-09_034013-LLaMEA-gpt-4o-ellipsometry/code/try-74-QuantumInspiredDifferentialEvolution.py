import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Initial crossover probability
        self.q_prob = 0.1  # Probability to use quantum-inspired move

    def quantum_move(self, individual, best, lb, ub):
        factor = np.random.uniform(0.5, 1.5)
        new_position = best + factor * (individual - best)
        return np.clip(new_position, lb, ub)

    def adaptive_mutation(self):
        return 0.5 + np.random.rand() * 0.3  # Adaptive mutation factor

    def adaptive_crossover(self, current_score, best_score):
        # Adapt CR based on the progress towards the best score
        return self.CR * (1 - current_score / (best_score + 1e-9))

    def quantum_crossover(self, individual, best, current_score, best_score):
        # Use adaptive crossover probability
        adaptive_CR = self.adaptive_crossover(current_score, best_score)
        return np.where(np.random.rand(self.dim) < adaptive_CR, best, individual)

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

                # Mutation with adaptive rate
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                self.F = self.adaptive_mutation()
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, lb, ub)

                # Quantum-inspired crossover with adaptive probability
                trial = self.quantum_crossover(mutant, global_best_position, scores[i], global_best_score)

                # Selection
                trial_score = func(trial)
                evaluations += 1
                if trial_score < scores[i]:
                    population[i] = trial
                    scores[i] = trial_score

                # Quantum-inspired move with a small probability
                if np.random.rand() < self.q_prob:
                    population[i] = self.quantum_move(population[i], global_best_position, lb, ub)

                # Update global best
                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = population[i].copy()

        return global_best_position, global_best_score