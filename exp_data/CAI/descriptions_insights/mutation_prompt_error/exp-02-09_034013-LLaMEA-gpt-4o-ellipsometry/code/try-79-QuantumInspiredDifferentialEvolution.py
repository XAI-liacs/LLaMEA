import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.init_population_size = 10 + int(2 * np.sqrt(dim))
        self.population_size = self.init_population_size
        self.F_min, self.F_max = 0.4, 0.9  # Dynamic differential weight boundaries
        self.CR_min, self.CR_max = 0.5, 0.9  # Adaptive crossover probability range
        self.q_prob = 0.1

    def levy_flight(self, lam=1.5):
        u = np.random.normal(0, 1, self.dim) * (np.random.gamma(1 + lam) * np.sin(np.pi * lam / 2) /
            (np.cos(np.pi * lam / 2) * lam * np.exp(np.log(np.pi / 2))))
        v = np.random.normal(0, 1, self.dim)
        z = u / (np.abs(v) ** (1 / lam))
        return z

    def adaptive_mutation(self, score, best_score):
        return self.F_min + (self.F_max - self.F_min) * (best_score - score) / best_score

    def adaptive_crossover(self, score, best_score):
        return self.CR_min + (self.CR_max - self.CR_min) * (best_score - score) / best_score

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
                self.F = self.adaptive_mutation(scores[i], global_best_score)
                mutant = population[a] + self.F * (population[b] - population[c]) + 0.1 * self.levy_flight()
                mutant = np.clip(mutant, lb, ub)

                self.CR = self.adaptive_crossover(scores[i], global_best_score)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])

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

            if evaluations % (self.budget // 10) == 0:
                self.population_size = max(5, int(self.init_population_size * (1 - evaluations / self.budget)))
                population = population[:self.population_size]
                scores = scores[:self.population_size]
                
        return global_best_position, global_best_score