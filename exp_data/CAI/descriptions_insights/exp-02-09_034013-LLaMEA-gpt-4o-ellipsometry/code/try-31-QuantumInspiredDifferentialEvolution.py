import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.initial_F = 0.5  # Initial Differential weight
        self.initial_CR = 0.9  # Initial Crossover probability
        self.q_prob = 0.1  # Probability to use quantum-inspired move
        self.elite_size = max(1, self.population_size // 10)  # Elite pool size
        self.F = self.initial_F
        self.CR = self.initial_CR

    def adapt_parameters(self, iter_num, max_iters):
        self.F = self.initial_F * (1 - iter_num / max_iters)
        self.CR = self.initial_CR * (iter_num / max_iters)

    def quantum_move(self, individual, best, lb, ub):
        factor = np.random.uniform(0.5, 1.5)
        new_position = best + factor * (individual - best)
        return np.clip(new_position, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        scores = np.full(self.population_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0
        max_iters = self.budget // self.population_size

        for iter_num in range(max_iters):
            self.adapt_parameters(iter_num, max_iters)

            elite_indices = np.argsort(scores)[:self.elite_size]
            elite_population = population[elite_indices]

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, lb, ub)

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                trial_score = func(trial)
                evaluations += 1
                if trial_score < scores[i]:
                    population[i] = trial
                    scores[i] = trial_score

                if np.random.rand() < self.q_prob:
                    population[i] = self.quantum_move(population[i], elite_population[np.random.randint(self.elite_size)], lb, ub)

                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = population[i].copy()

        return global_best_position, global_best_score