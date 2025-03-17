import numpy as np

class HSBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory_size = 12  # Increased memory size for more diversity
        bees_count = 6  # Adjusted bees count for better sampling
        harmony_memory = np.random.uniform(lb, ub, (harmony_memory_size, self.dim))
        harmony_memory_fitness = np.array([func(sol) for sol in harmony_memory])
        evals = harmony_memory_size

        while evals < self.budget:
            for bee in range(bees_count):
                if evals >= self.budget:
                    break

                # Adaptive memory consideration
                selected_idx = np.random.choice(harmony_memory_size, p=(1 / harmony_memory_fitness) / np.sum(1 / harmony_memory_fitness))
                new_solution = np.copy(harmony_memory[selected_idx])

                # Adaptive pitch adjustment with stochastic component
                fitness_std = np.std(harmony_memory_fitness)
                fitness_mean = np.mean(harmony_memory_fitness)
                pitch_adj_rate = 0.3 + 0.7 * (fitness_std / (fitness_mean + 1e-8))  # Adjusted pitch rate
                stochastic_factor = np.random.rand() * pitch_adj_rate
                for i in range(self.dim):
                    if np.random.rand() < pitch_adj_rate:
                        adjustment = np.random.uniform(-1, 1) * (ub[i] - lb[i]) * stochastic_factor
                        new_solution[i] += adjustment

                # Enhanced neighborhood search
                if np.random.rand() < 0.6:  # Increased probability
                    neighbor = np.random.randint(0, harmony_memory_size)
                    new_solution = 0.5 * (new_solution + harmony_memory[neighbor]) + 0.5 * np.random.uniform(lb, ub, self.dim)

                new_solution = np.clip(new_solution, lb, ub)
                
                new_fitness = func(new_solution)
                evals += 1

                # Replace using a probabilistic acceptance criterion
                worst_idx = np.argmax(harmony_memory_fitness)
                acceptance_probability = 1 / (1 + np.exp((new_fitness - harmony_memory_fitness[worst_idx]) / (fitness_std + 1e-8)))
                if new_fitness < harmony_memory_fitness[worst_idx] or np.random.rand() < acceptance_probability:
                    harmony_memory[worst_idx] = new_solution
                    harmony_memory_fitness[worst_idx] = new_fitness

        best_idx = np.argmin(harmony_memory_fitness)
        return harmony_memory[best_idx]