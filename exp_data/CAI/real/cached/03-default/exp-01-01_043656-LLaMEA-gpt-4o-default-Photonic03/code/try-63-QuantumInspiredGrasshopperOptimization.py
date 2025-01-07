import numpy as np

class QuantumInspiredGrasshopperOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.r_max = 1.0
        self.r_min = 0.00001
        self.quantum_factor_initial = 0.5
        self.quantum_factor_final = 0.1

    def quantum_position_update(self, position, best_position, eval_count):
        r = self.r_max - (self.r_max - self.r_min) * (eval_count / self.budget)
        quantum_factor = self.quantum_factor_initial * (1 - eval_count / self.budget) + self.quantum_factor_final * (eval_count / self.budget)
        noise = np.random.rand(self.dim)
        new_position = position + quantum_factor * r * (best_position - position) * noise
        return new_position

    def grasshopper_movement(self, position, population, eval_count):
        s = np.zeros(self.dim)
        r = self.r_max - (self.r_max - self.r_min) * (eval_count / self.budget)
        for j in range(self.population_size):
            if not np.array_equal(position, population[j]):
                dist = np.linalg.norm(position - population[j])
                s += ((population[j] - position) / (dist + np.finfo(float).eps)) * np.exp(-dist / r)
        return s

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in pop])
        best_position = pop[np.argmin(fitness)]
        best_value = fitness.min()
        
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                s_i = self.grasshopper_movement(pop[i], pop, eval_count)
                quantum_position = self.quantum_position_update(pop[i], best_position, eval_count)
                candidate_position = pop[i] + s_i + quantum_position
                candidate_position = np.clip(candidate_position, bounds[:, 0], bounds[:, 1])
                
                candidate_value = func(candidate_position)
                eval_count += 1
                if candidate_value < fitness[i]:
                    pop[i] = candidate_position
                    fitness[i] = candidate_value
                    if candidate_value < best_value:
                        best_position = candidate_position
                        best_value = candidate_value
                
                if eval_count >= self.budget:
                    break

        return best_position