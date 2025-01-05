import numpy as np

class QAMPDE:
    def __init__(self, budget, dim, num_populations=5, population_size=10, F=0.5, CR=0.9, quantum_prob=0.2, diversity_threshold=0.1):
        self.budget = budget
        self.dim = dim
        self.num_populations = num_populations
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.quantum_prob = quantum_prob
        self.evaluations = 0
        self.diversity_threshold = diversity_threshold

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')

        populations = [self.initialize_population(lb, ub) for _ in range(self.num_populations)]

        while self.evaluations < self.budget:
            for pop_id in range(self.num_populations):
                diversity = self.calculate_diversity(populations[pop_id])
                dynamic_quantum_prob = self.quantum_prob * (1 + (self.diversity_threshold - diversity))
                
                for i in range(self.population_size):
                    target = populations[pop_id][i]
                    
                    # Mutation
                    a, b, c = populations[pop_id][np.random.choice(self.population_size, 3, replace=False)]
                    mutant = np.clip(a + self.F * (b - c), lb, ub)
                    
                    # Crossover
                    trial = np.array([mutant[j] if np.random.rand() < self.CR else target[j] for j in range(self.dim)])
                    
                    # Apply dynamic quantum-inspired perturbation
                    if np.random.rand() < dynamic_quantum_prob:
                        trial = self.quantum_perturbation(trial, lb, ub)
                    
                    value = func(trial)
                    self.evaluations += 1

                    if value < func(target):
                        populations[pop_id][i] = trial

                    if value < best_global_value:
                        best_global_value = value
                        best_global_position = trial

                    if self.evaluations >= self.budget:
                        break

                self.evolve_population(populations[pop_id], lb, ub)

                if self.evaluations >= self.budget:
                    break

        return best_global_position

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def evolve_population(self, population, lb, ub):
        for i in range(self.population_size):
            if np.random.rand() < 0.1:
                mutation = (np.random.rand(self.dim) - 0.5) * 0.1 * (ub - lb)
                population[i] = np.clip(population[i] + mutation, lb, ub)
        
        for i in range(0, self.population_size, 2):
            if i+1 < self.population_size and np.random.rand() < 0.2:
                crossover_point = np.random.randint(1, self.dim)
                population[i][:crossover_point], population[i+1][:crossover_point] = (
                    population[i+1][:crossover_point].copy(), population[i][:crossover_point].copy())

    def quantum_perturbation(self, position, lb, ub):
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.05
        return np.clip(q_position, lb, ub)

    def calculate_diversity(self, population):
        centroid = np.mean(population, axis=0)
        diversity = np.mean(np.linalg.norm(population - centroid, axis=1))
        return diversity