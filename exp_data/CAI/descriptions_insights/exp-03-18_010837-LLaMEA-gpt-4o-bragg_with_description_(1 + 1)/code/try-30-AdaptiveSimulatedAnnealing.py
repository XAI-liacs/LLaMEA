import numpy as np

class AdaptiveSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.temperature = 1.0
        self.cooling_rate = 0.95
        self.initial_step_size = 0.1
        self.step_size = self.initial_step_size
        self.min_step_size = 1e-9
        self.elite_fraction = 0.1

    def _acceptance_probability(self, cost, new_cost, temperature):
        if new_cost < cost:
            return 1.0
        return np.exp((cost - new_cost) / temperature)

    def __call__(self, func):
        best_solution = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        best_cost = func(best_solution)
        current_solution = np.copy(best_solution)
        current_cost = best_cost
        elite_solutions = [best_solution]

        for _ in range(self.budget):
            # Introduce Levy flight for exploration
            levy_flight = np.random.standard_cauchy(self.dim)
            candidate_solution = current_solution + levy_flight * self.step_size
            candidate_solution = np.clip(candidate_solution, func.bounds.lb, func.bounds.ub)
            candidate_cost = func(candidate_solution)

            if self._acceptance_probability(current_cost, candidate_cost, self.temperature) > np.random.rand():
                current_solution = candidate_solution
                current_cost = candidate_cost

                if candidate_cost < best_cost:
                    best_solution = candidate_solution
                    best_cost = candidate_cost
                    elite_solutions.append(best_solution)
                    elite_solutions.sort(key=func)
                    elite_solutions = elite_solutions[:int(self.elite_fraction * len(elite_solutions) + 1)]

            self.temperature *= self.cooling_rate
            
            # Adaptive step size with dynamic learning rate
            if candidate_cost < current_cost:
                self.step_size = min(self.step_size * 1.1, self.initial_step_size)
            else:
                self.step_size = max(self.step_size * 0.9, self.min_step_size)

            # Introduce elite solution influence
            if elite_solutions:
                influence = np.mean(elite_solutions, axis=0)
                current_solution = (current_solution + influence) / 2
            
            # Dynamically adapt elite fraction based on convergence speed
            self.elite_fraction = min(0.5, max(0.1, self.elite_fraction * (1 + 0.01 * (best_cost - current_cost) / self.budget)))

        return best_solution