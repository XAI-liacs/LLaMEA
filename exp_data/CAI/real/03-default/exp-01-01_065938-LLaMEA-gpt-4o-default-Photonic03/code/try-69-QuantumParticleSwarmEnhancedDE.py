import numpy as np

class QuantumParticleSwarmEnhancedDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F_min, self.F_max = 0.5, 0.9
        self.CR_min, self.CR_max = 0.6, 0.9
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        velocity = np.random.rand(self.population_size, self.dim)
        fitness = np.array([func(x) for x in population])
        personal_best_pos = np.copy(population)
        personal_best_fitness = np.copy(fitness)
        best_global_idx = np.argmin(fitness)
        best_global_pos = population[best_global_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Update velocity and position
                velocity[i] = (self.inertia_weight * velocity[i] +
                               self.cognitive_coeff * np.random.rand(self.dim) * (personal_best_pos[i] - population[i]) +
                               self.social_coeff * np.random.rand(self.dim) * (best_global_pos - population[i]))
                
                population[i] += velocity[i]
                population[i] = np.clip(population[i], lb, ub)

                # Quantum-inspired mutation
                quantum_mutation = population[i] + np.random.uniform(-1, 1, self.dim) * (best_global_pos - population[i])
                quantum_mutation = np.clip(quantum_mutation, lb, ub)

                F = np.random.uniform(self.F_min, self.F_max)
                CR = np.random.uniform(self.CR_min, self.CR_max)
                indices = np.random.choice(range(self.population_size), 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = x0 + F * (x1 - x2)
                mutant = np.clip(mutant, lb, ub)

                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, quantum_mutation)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < personal_best_fitness[i]:
                        personal_best_pos[i] = trial
                        personal_best_fitness[i] = trial_fitness

                        if trial_fitness < fitness[best_global_idx]:
                            best_global_idx = i
                            best_global_pos = trial

        return best_global_pos