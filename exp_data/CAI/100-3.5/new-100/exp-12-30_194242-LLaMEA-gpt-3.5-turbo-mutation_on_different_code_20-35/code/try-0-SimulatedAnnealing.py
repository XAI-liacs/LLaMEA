import numpy as np

class SimulatedAnnealing:
    def __init__(self, budget=10000, dim=10, initial_temp=1.0, cooling_factor=0.95):
        self.budget = budget
        self.dim = dim
        self.initial_temp = initial_temp
        self.cooling_factor = cooling_factor
        self.f_opt = np.Inf
        self.x_opt = None

    def acceptance_probability(self, current_cost, new_cost, temp):
        if new_cost < current_cost:
            return 1.0
        return np.exp((current_cost - new_cost) / temp)

    def __call__(self, func):
        current_state = np.random.uniform(func.bounds.lb, func.bounds.ub)
        current_cost = func(current_state)
        temp = self.initial_temp

        for _ in range(self.budget):
            new_state = current_state + np.random.normal(0, 0.1, self.dim)
            new_state = np.clip(new_state, func.bounds.lb, func.bounds.ub)
            new_cost = func(new_state)

            if self.acceptance_probability(current_cost, new_cost, temp) > np.random.random():
                current_state = new_state
                current_cost = new_cost

            if current_cost < self.f_opt:
                self.f_opt = current_cost
                self.x_opt = current_state

            temp *= self.cooling_factor

        return self.f_opt, self.x_opt