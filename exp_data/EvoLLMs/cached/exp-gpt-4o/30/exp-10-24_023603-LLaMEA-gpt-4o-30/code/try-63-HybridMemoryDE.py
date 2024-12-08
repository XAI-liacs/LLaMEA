import numpy as np

class HybridMemoryDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.memory_size = 5
        self.cross_prob = 0.9
        self.F = 0.5
        self.epsilon = 0.01
        self.probability = 0.3
        
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.memory = {
            "F": np.full(self.memory_size, self.F),
            "CR": np.full(self.memory_size, self.cross_prob)
        }
        self.memory_index = 0

    def update_memory(self, F, CR):
        self.memory["F"][self.memory_index] = F
        self.memory["CR"][self.memory_index] = CR
        self.memory_index = (self.memory_index + 1) % self.memory_size

    def __call__(self, func):
        eval_count = 0
        best_fitness = np.inf
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 5, replace=False)
                a, b, c, d, e = self.population[indices]
                
                if np.random.rand() < self.probability:
                    # Multi-strategy mutation incorporating both local and global search dynamics
                    if np.random.rand() < 0.5:
                        F = np.random.choice(self.memory["F"]) + np.random.rand() * self.epsilon
                        mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                    else:
                        F = np.random.choice(self.memory["F"]) + np.random.rand() * self.epsilon
                        mutant = np.clip(a + F * (b - c + d - e), self.lower_bound, self.upper_bound)
                else:
                    # Hybrid mutation integrating random and historical memory influences
                    hybrid_factor = (b + c) / 2
                    F = np.random.choice(self.memory["F"]) + np.random.rand() * self.epsilon
                    mutant = np.clip(a + F * (hybrid_factor - e), self.lower_bound, self.upper_bound)
                
                # Dynamic Crossover based on fitness improvement
                CR = np.random.choice(self.memory["CR"]) + np.random.rand() * self.epsilon
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])

                # Selection
                trial_fitness = func(trial)
                eval_count += 1

                if eval_count >= self.budget:
                    break

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        self.update_memory(F, CR)
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]