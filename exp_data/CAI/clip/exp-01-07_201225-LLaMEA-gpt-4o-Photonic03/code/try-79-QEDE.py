import numpy as np

class QEDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(20, 5 + dim)
        self.population = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, np.inf)
        self.best_solution = None
        self.best_fitness = np.inf
        self.elitism_rate = 0.2  # New: Elitism rate

    def __call__(self, func):
        bounds = np.vstack((func.bounds.lb, func.bounds.ub)).T
        evaluations = 0

        def decode_position(position):
            return bounds[:, 0] + ((bounds[:, 1] - bounds[:, 0]) * position)

        def evaluate_position(decoded_position):
            nonlocal evaluations
            fitness = np.array([func(ind) for ind in decoded_position])
            evaluations += len(decoded_position)
            return fitness

        def quantum_mutation(base, best, r1, r2):
            phi = np.random.rand(self.dim)
            perturbation = 0.1 * np.exp(-evaluations / (0.5 * self.budget))
            scaling_factor = 0.9 * (1 - evaluations / self.budget)
            return base + phi * (best - base) + (1 - phi) * (r1 - r2) * scaling_factor \
                   + phi * (np.random.rand(self.dim) - 0.5) * perturbation

        def quantum_crossover(target, mutant):
            rand_vec = np.random.rand(self.dim)
            return np.where(rand_vec < 0.5, target, mutant)

        while evaluations < self.budget:
            decoded_population = decode_position(self.population)
            fitness = evaluate_position(decoded_population)

            for i in range(self.population_size):
                if fitness[i] < self.fitness[i]:
                    self.fitness[i] = fitness[i]
                    if fitness[i] < self.best_fitness:
                        self.best_fitness = fitness[i]
                        self.best_solution = self.population[i]

            num_elites = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(self.fitness)[:num_elites]
            elites = self.population[elite_indices]

            for i in range(self.population_size):
                if i in elite_indices:
                    continue

                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = quantum_mutation(a, self.best_solution, b, c)
                mutant = np.clip(mutant, 0, 1)
                trial = quantum_crossover(self.population[i], mutant)

                trial_decoded = decode_position(trial)
                trial_fitness = func(trial_decoded)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial

            # Incorporate diversity mechanism
            if evaluations < self.budget:
                diversity_threshold = 0.05 * (1 - evaluations / self.budget)
                for i in range(self.population_size):
                    diversity = np.mean([np.linalg.norm(self.population[i] - self.population[j]) for j in range(self.population_size) if j != i])
                    if diversity < diversity_threshold:
                        self.population[i] = np.random.rand(self.dim)

        best_solution = decode_position(self.best_solution)
        return best_solution, self.best_fitness