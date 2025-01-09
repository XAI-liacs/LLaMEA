import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.pso_max_iter = 100
        self.sa_max_iter = 50
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def pso_init():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

        def sa_init():
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

        def evaluate_solution(solution):
            return func(solution)

        def sa_acceptance_probability(curr_cost, new_cost, temperature):
            if new_cost < curr_cost:
                return 1
            return np.exp((curr_cost - new_cost) / temperature)

        def sa_optimize(initial_solution):
            curr_solution = initial_solution
            curr_cost = evaluate_solution(curr_solution)
            best_solution = np.copy(curr_solution)
            best_cost = curr_cost
            temperature = 1.0
            for _ in range(self.sa_max_iter):
                new_solution = curr_solution + np.random.normal(0, 0.1, self.dim)
                new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
                new_cost = evaluate_solution(new_solution)
                if sa_acceptance_probability(curr_cost, new_cost, temperature) > np.random.rand():
                    curr_solution = new_solution
                    curr_cost = new_cost
                    if new_cost < best_cost:
                        best_solution = np.copy(new_solution)
                        best_cost = new_cost
                temperature *= 0.95
            return best_solution

        particles = pso_init()
        for _ in range(self.pso_max_iter):
            for i in range(self.population_size):
                particles[i] = sa_optimize(particles[i])
            best_particle = min(particles, key=lambda x: evaluate_solution(x))
            if evaluate_solution(best_particle) < evaluate_solution(global_best_particle):
                global_best_particle = best_particle

        return evaluate_solution(global_best_particle)