import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.q_prob = 0.1  # Probability to use quantum-inspired move

    def levy_flight(self, individual, lb, ub):
        beta = 1.5
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                   (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.abs(v) ** (1 / beta)
        new_position = individual + 0.01 * step * (individual - lb)
        return np.clip(new_position, lb, ub)

    def dynamic_crossover(self, iteration):
        return self.CR * (1 - 0.5 * (iteration / self.budget))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        scores = np.full(self.population_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0
        iteration = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Mutation with Lévy flight
                self.F = self.adaptive_mutation()
                mutant = self.levy_flight(population[i], lb, ub)

                # Dynamic crossover
                CR_dynamic = self.dynamic_crossover(evaluations)
                trial = np.where(np.random.rand(self.dim) < CR_dynamic, mutant, population[i])

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

            iteration += 1

        return global_best_position, global_best_score