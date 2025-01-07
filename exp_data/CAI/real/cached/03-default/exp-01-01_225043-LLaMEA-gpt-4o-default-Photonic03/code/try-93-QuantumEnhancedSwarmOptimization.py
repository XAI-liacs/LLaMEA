import numpy as np

class QuantumEnhancedSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.individuals = np.random.uniform(size=(self.population_size, dim))
        self.velocities = np.random.uniform(size=(self.population_size, dim)) * 0.1
        self.personal_best = self.individuals.copy()
        self.global_best = None
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best_fitness = np.inf
        self.fitness_evaluations = 0

    def quantum_mutation(self, L, scale):
        tau = scale / np.sqrt(self.dim)
        return np.random.normal(0, tau, size=L)

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]
        
        inertia_weight = lambda evals: 0.9 - 0.5 * (evals / self.budget)
        cognitive_component = lambda evals: 1.5 + 1.0 * (evals / self.budget)
        social_component = lambda evals: 1.5 - 1.0 * (evals / self.budget)

        while self.fitness_evaluations < self.budget:
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break
                
                fitness = func(self.individuals[i])
                self.fitness_evaluations += 1

                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = self.individuals[i].copy()

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = self.individuals[i].copy()
            
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break
                
                inertia = inertia_weight(self.fitness_evaluations)
                cognitive = cognitive_component(self.fitness_evaluations)
                social = social_component(self.fitness_evaluations)

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocities[i] = (inertia * self.velocities[i] +
                                      cognitive * r1 * (self.personal_best[i] - self.individuals[i]) +
                                      social * r2 * (self.global_best - self.individuals[i]))
                
                self.individuals[i] += self.velocities[i]
                self.individuals[i] = np.clip(self.individuals[i], lower_bound, upper_bound)

                quantum_step = self.quantum_mutation(self.dim, self.fitness_evaluations / self.budget)
                if np.random.rand() < 0.2:
                    self.individuals[i] += quantum_step
                    self.individuals[i] = np.clip(self.individuals[i], lower_bound, upper_bound)

        return self.global_best