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
        self.momentum_factor = 0.9

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
        prev_solution = np.zeros(self.dim)

        for _ in range(self.budget):
            levy_flight = np.random.standard_cauchy(self.dim) * (1 + 0.1 * self.temperature) 
            candidate_solution = current_solution + levy_flight * self.step_size
            candidate_solution = np.clip(candidate_solution, func.bounds.lb, func.bounds.ub)
            candidate_cost = func(candidate_solution)

            if self._acceptance_probability(current_cost, candidate_cost, self.temperature) > np.random.rand():
                prev_solution = current_solution
                current_solution = candidate_solution
                current_cost = candidate_cost

                if candidate_cost < best_cost:
                    best_solution = candidate_solution
                    best_cost = candidate_cost
                    elite_solutions.append(best_solution)
                    elite_solutions = sorted(elite_solutions, key=func)[:int(self.elite_fraction * len(elite_solutions) + 1)]

            self.temperature *= self.cooling_rate

            if candidate_cost < current_cost:
                self.step_size = min(self.step_size * 1.1, self.initial_step_size)
            else:
                self.step_size = max(self.step_size * 0.9 * (1 + self.temperature), self.min_step_size)

            if elite_solutions:
                influence = np.mean(elite_solutions, axis=0)
                influence_factor = 0.5 + 0.5 * (1 - self.temperature)  # Adaptive influence factor
                current_solution = (current_solution + influence_factor * influence) / (1 + influence_factor)

            self.momentum_factor = 0.9 * (1 - self.temperature)
            current_solution += self.momentum_factor * (current_solution - prev_solution) * self.temperature

            if np.random.rand() < 0.01 + 0.04 * (1 - self.temperature):
                current_solution = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
                current_cost = func(current_solution)

            self.step_size *= (1 + 0.01 * self.temperature)

        return best_solution