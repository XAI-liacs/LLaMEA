import numpy as np

class HSAPA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.low_bound = -5.0
        self.up_bound = 5.0
        self.HM_size = 20
        self.HMCR = 0.7
        self.PAR_min = 0.3
        self.PAR_max = 0.9
        self.bw_min = 0.01
        self.bw_max = 1.0
        self.adaptive_rate = 0.1

    def harmony_search(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(self.low_bound, self.up_bound, (self.HM_size, self.dim))

        def adjust_parameter(value, min_val, max_val, rate):
            return max(min_val, min(max_val, value * (1 + np.random.uniform(-rate, rate)))

        def evaluate_solution(solution):
            return func(solution)

        def improvise_new_solution():
            new_solution = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.uniform(0, 1) < self.HMCR:
                    new_solution[i] = self.harmony_memory[np.random.randint(self.HM_size)][i]
                else:
                    new_solution[i] = np.random.uniform(self.low_bound, self.up_bound)
                    if np.random.uniform(0, 1) < self.PAR:
                        new_solution[i] += np.random.uniform(-1, 1) * self.bandwidth
            return new_solution

        self.harmony_memory = initialize_harmony_memory()
        self.PAR = self.PAR_max
        self.bandwidth = self.bw_max

        for _ in range(self.budget):
            new_solution = improvise_new_solution()
            new_solution_fitness = evaluate_solution(new_solution)

            if new_solution_fitness < min(self.fitness_values):
                min_index = np.argmin(self.fitness_values)
                self.harmony_memory[min_index] = new_solution
                self.fitness_values[min_index] = new_solution_fitness

            self.PAR = adjust_parameter(self.PAR, self.PAR_min, self.PAR_max, self.adaptive_rate)
            self.bandwidth = adjust_parameter(self.bandwidth, self.bw_min, self.bw_max, self.adaptive_rate)

            if np.random.uniform() < 0.1:  # Introduce dynamic bandwidth adjustment
                self.bandwidth = np.clip(self.bandwidth * np.random.choice([0.9, 1.1]), self.bw_min, self.bw_max)

        best_index = np.argmin(self.fitness_values)
        return self.harmony_memory[best_index]

    def __call__(self, func):
        self.fitness_values = [np.inf] * self.HM_size
        return self.harmony_search(func)