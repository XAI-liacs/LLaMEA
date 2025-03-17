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

        while evals < self.budget:
            for bee in range(bees_count):
                if evals >= self.budget:
                    break

                # Adaptive memory consideration
                probabilities = harmony_memory_fitness / harmony_memory_fitness.sum()
                selected_idx = np.random.choice(harmony_memory_size, p=probabilities)
                new_solution = np.copy(harmony_memory[selected_idx])

                # Modified pitch adjustment with controlled stochastic factor
                fitness_std = np.std(harmony_memory_fitness)
                fitness_mean = np.mean(harmony_memory_fitness)
                pitch_adj_rate = 0.3 + 0.7 * (fitness_std / (fitness_mean + 1e-8))
                for i in range(self.dim):
                    if np.random.rand() < pitch_adj_rate:
                        adjustment = np.random.uniform(-0.5, 0.5) * (ub[i] - lb[i]) * pitch_adj_rate
                        new_solution[i] += adjustment

                # Adaptive neighborhood search
                if np.random.rand() < 0.5:
                    neighbor = np.random.randint(0, harmony_memory_size)
                    new_solution = 0.7 * new_solution + 0.3 * harmony_memory[neighbor]

                new_solution = np.clip(new_solution, lb, ub)

                new_fitness = func(new_solution)
                evals += 1

                # Enhanced probabilistic acceptance criterion
                worst_idx = np.argmax(harmony_memory_fitness)
                acceptance_probability = np.exp(-abs(new_fitness - harmony_memory_fitness[worst_idx]) / (fitness_std + 1e-8))
                if new_fitness < harmony_memory_fitness[worst_idx] or np.random.rand() < acceptance_probability:
                    harmony_memory[worst_idx] = new_solution
                    harmony_memory_fitness[worst_idx] = new_fitness

        best_idx = np.argmin(harmony_memory_fitness)
        return harmony_memory[best_idx]