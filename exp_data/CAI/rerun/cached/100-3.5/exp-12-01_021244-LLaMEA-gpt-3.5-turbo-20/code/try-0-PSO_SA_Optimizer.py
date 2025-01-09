import numpy as np

class PSO_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 20
        self.max_iterations = budget // self.swarm_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.inertia_weight = 0.5
        self.c1 = 2.0
        self.c2 = 2.0
        self.initial_temperature = 100.0
        self.final_temperature = 0.1
        self.alpha = 0.85

    def __call__(self, func):
        def random_solution():
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

        def within_bounds(solution):
            return np.clip(solution, self.lower_bound, self.upper_bound)

        def objective(solution):
            return func(solution)

        def pso():
            swarm = np.array([random_solution() for _ in range(self.swarm_size)])
            swarm_best = swarm[np.argmin([objective(sol) for sol in swarm])]
            global_best = swarm_best.copy()

            for _ in range(self.max_iterations):
                for i in range(self.swarm_size):
                    velocity = self.inertia_weight * velocity + self.c1 * np.random.rand(self.dim) * (swarm_best - swarm[i]) + self.c2 * np.random.rand(self.dim) * (global_best - swarm[i])
                    candidate_position = swarm[i] + velocity
                    swarm[i] = within_bounds(candidate_position)
                    if objective(swarm[i]) < objective(swarm_best):
                        swarm_best = swarm[i]
                    if objective(swarm[i]) < objective(global_best):
                        global_best = swarm[i]
            return global_best

        def sa(initial_solution):
            current_solution = initial_solution
            current_cost = objective(current_solution)
            temperature = self.initial_temperature

            for _ in range(self.max_iterations):
                candidate_solution = current_solution + np.random.normal(0, temperature, self.dim)
                candidate_solution = within_bounds(candidate_solution)
                candidate_cost = objective(candidate_solution)
                
                if candidate_cost < current_cost or np.random.rand() < np.exp((current_cost - candidate_cost) / temperature):
                    current_solution = candidate_solution
                    current_cost = candidate_cost

                temperature *= self.alpha
                if temperature < self.final_temperature:
                    break

            return current_solution

        best_solution = random_solution()
        for _ in range(self.budget // self.swarm_size):
            pso_solution = pso()
            sa_solution = sa(pso_solution)
            if objective(sa_solution) < objective(best_solution):
                best_solution = sa_solution

        return best_solution