import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.F = 0.5
        self.CR = 0.9
        self.q_prob = 0.1
        self.dynamic_population = True

    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v)**(1 / beta)
        return step

    def quantum_move(self, individual, best, lb, ub):
        step = self.levy_flight(self.dim) if np.random.rand() < 0.5 else np.random.uniform(0.5, 1.5)
        new_position = best + step * (individual - best)
        return np.clip(new_position, lb, ub)

    def adaptive_mutation(self):
        return 0.5 + np.random.rand() * 0.3

    def quantum_crossover(self, individual, best):
        return np.where(np.random.rand(self.dim) < self.q_prob, best, individual)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        scores = np.full(self.population_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0

        while evaluations < self.budget:
            if self.dynamic_population:
                self.population_size = 10 + int(2 * np.sqrt(self.dim)) - int(evaluations / self.budget * 5)
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