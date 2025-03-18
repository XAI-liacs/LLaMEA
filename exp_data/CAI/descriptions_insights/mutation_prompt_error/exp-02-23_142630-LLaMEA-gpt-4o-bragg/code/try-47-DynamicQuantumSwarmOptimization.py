import numpy as np

class DynamicQuantumSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.alpha = 0.5  # Cognitive weight
        self.beta = 0.3   # Social weight
        self.gamma = 0.2  # Inertia weight
        self.history = []

    def quantum_update(self, particle, personal_best, global_best):
        q = np.random.rand(self.dim)
        c1 = self.alpha * np.random.rand(self.dim) * (personal_best - particle)
        c2 = self.beta * np.random.rand(self.dim) * (global_best - particle)
        inertia = self.gamma * q * (global_best - personal_best)
        new_particle = particle + c1 + c2 + inertia
        return np.clip(new_particle, self.bounds.lb, self.bounds.ub)

    def __call__(self, func):
        self.bounds = func.bounds
        self.population_size = min(max(10, int(self.budget / (self.dim * 3))), self.population_size)
        population = np.random.uniform(self.bounds.lb, self.bounds.ub, (self.population_size, self.dim))
        personal_bests = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in personal_bests])
        global_best_idx = np.argmin(personal_best_scores)
        global_best = personal_bests[global_best_idx]

        self.history.extend(personal_best_scores)

        evaluations = self.population_size
        while evaluations < self.budget:
            new_population = []
            for i in range(self.population_size):
                new_particle = self.quantum_update(population[i], personal_bests[i], global_best)
                new_score = func(new_particle)
                evaluations += 1

                if new_score < personal_best_scores[i]:
                    personal_bests[i] = new_particle
                    personal_best_scores[i] = new_score

                new_population.append(new_particle)

                if new_score < personal_best_scores[global_best_idx]:
                    global_best = new_particle
                    global_best_idx = i

            population = np.array(new_population)
            self.history.extend(personal_best_scores)

        return global_best, personal_best_scores[global_best_idx], self.history