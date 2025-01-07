import numpy as np

class CCPSO_Optimizer:
    def __init__(self, budget, dim, subcomponents=5):
        self.budget = budget
        self.dim = dim
        self.subcomponents = subcomponents
        self.population_size = 20
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.7
        self.sub_dim = self.dim // self.subcomponents
        self.eval_budget_per_component = budget // self.subcomponents

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)
        
        # Initialize the swarm for each subcomponent
        positions = [np.random.uniform(lb, ub, (self.population_size, self.sub_dim)) for _ in range(self.subcomponents)]
        velocities = [np.random.uniform(-1, 1, (self.population_size, self.sub_dim)) for _ in range(self.subcomponents)]
        personal_best_positions = [np.copy(pos) for pos in positions]
        personal_best_values = [np.array([func(np.concatenate([p if j == i else positions[j][0] 
                                           for j in range(self.subcomponents)])) for p in pos]) 
                                for i, pos in enumerate(personal_best_positions)]
        
        global_best_position = np.concatenate([pb[np.argmin(pv)] for pb, pv in zip(personal_best_positions, personal_best_values)])
        global_best_value = np.min([np.min(pv) for pv in personal_best_values])
        
        evaluations = self.population_size * self.subcomponents

        for component in range(self.subcomponents):
            while evaluations < self.eval_budget_per_component * (component + 1):
                for i in range(self.population_size):
                    # Update particle velocity and position for the current subcomponent
                    r1, r2 = np.random.rand(2)
                    velocities[component][i] = (self.w * velocities[component][i] +
                                                self.c1 * r1 * (personal_best_positions[component][i] - positions[component][i]) +
                                                self.c2 * r2 * (global_best_position[component*self.sub_dim:(component+1)*self.sub_dim] - positions[component][i]))
                    positions[component][i] += velocities[component][i]
                    positions[component][i] = np.clip(positions[component][i], lb, ub)

                    # Evaluate
                    candidate_solution = np.concatenate([positions[component][i] if j == component else positions[j][0] 
                                                         for j in range(self.subcomponents)])
                    current_value = func(candidate_solution)
                    evaluations += 1

                    # Update personal best for this subcomponent
                    if current_value < personal_best_values[component][i]:
                        personal_best_positions[component][i] = positions[component][i]
                        personal_best_values[component][i] = current_value

                    # Update global best
                    if current_value < global_best_value:
                        global_best_position = candidate_solution
                        global_best_value = current_value

        return global_best_position, global_best_value