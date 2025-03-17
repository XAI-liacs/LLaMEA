import numpy as np

class HSBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory_size = 10
        bees_count = 5
        harmony_memory = np.random.uniform(lb, ub, (harmony_memory_size, self.dim))
        harmony_memory_fitness = np.array([func(sol) for sol in harmony_memory])
        evals = harmony_memory_size

        # Identify elite solution
        elite_idx = np.argmin(harmony_memory_fitness)
        elite_solution = harmony_memory[elite_idx]

        while evals < self.budget:
            for bee in range(bees_count):
                if evals >= self.budget:
                    break

                selected_idx = np.random.choice(harmony_memory_size)
                new_solution = np.copy(harmony_memory[selected_idx])

                fitness_std = np.std(harmony_memory_fitness)
                fitness_mean = np.mean(harmony_memory_fitness)
                
                # Introduce self-adaptive scaling factor
                scaling_factor = 0.5 + 0.5 * (fitness_std / (fitness_mean + 1e-8))
                stochastic_factor = np.random.rand() * scaling_factor
                for i in range(self.dim):
                    if np.random.rand() < scaling_factor:
                        adjustment = np.random.uniform(-1, 1) * (ub[i] - lb[i]) * stochastic_factor
                        new_solution[i] += adjustment

                if np.random.rand() < 0.5:
                    neighbor = np.random.randint(0, harmony_memory_size)
                    new_solution = 0.5 * (new_solution + harmony_memory[neighbor])

                new_solution = np.clip(new_solution, lb, ub)

                new_fitness = func(new_solution)
                evals += 1

                worst_idx = np.argmax(harmony_memory_fitness)
                acceptance_probability = 1 / (1 + np.exp((new_fitness - harmony_memory_fitness[worst_idx]) / (fitness_std + 1e-8)))
                
                # Preserve elite solution
                if new_fitness < harmony_memory_fitness[worst_idx] or np.random.rand() < acceptance_probability:
                    harmony_memory[worst_idx] = new_solution
                    harmony_memory_fitness[worst_idx] = new_fitness
                    if new_fitness < harmony_memory_fitness[elite_idx]:
                        elite_solution = new_solution

        return elite_solution