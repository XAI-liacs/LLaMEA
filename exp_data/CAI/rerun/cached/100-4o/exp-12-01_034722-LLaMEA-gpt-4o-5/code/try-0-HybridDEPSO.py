import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.de_cross_prob = 0.9
        self.de_diff_weight = 0.8
        self.pso_inertia_weight = 0.5
        self.pso_cognitive_weight = 1.5
        self.pso_social_weight = 1.5

    def __call__(self, func):
        np.random.seed(0)
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, pop)
        personal_best_positions = np.copy(pop)
        personal_best_fitness = np.copy(fitness)
        global_best_position = pop[np.argmin(fitness)]
        global_best_fitness = np.min(fitness)
        
        evaluations = self.population_size
        while evaluations < self.budget:
            # Differential Evolution Mutation and Crossover
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.de_diff_weight * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.de_cross_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < personal_best_fitness[i]:
                        personal_best_positions[i] = trial
                        personal_best_fitness[i] = trial_fitness
            
            # Particle Swarm Optimization Update
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.pso_inertia_weight * velocities[i] +
                                 self.pso_cognitive_weight * r1 * (personal_best_positions[i] - pop[i]) +
                                 self.pso_social_weight * r2 * (global_best_position - pop[i]))
                pop[i] = np.clip(pop[i] + velocities[i], self.lower_bound, self.upper_bound)
                current_fitness = func(pop[i])
                evaluations += 1
                if current_fitness < fitness[i]:
                    fitness[i] = current_fitness
                    if current_fitness < personal_best_fitness[i]:
                        personal_best_positions[i] = pop[i]
                        personal_best_fitness[i] = current_fitness
                        if current_fitness < global_best_fitness:
                            global_best_position = pop[i]
                            global_best_fitness = current_fitness
        
        return global_best_position, global_best_fitness