import numpy as np
from scipy.optimize import minimize

class WDAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.history = []

    def wavelet_transform(self, candidate):
        transformed = np.fft.fft(candidate)
        low_pass = np.where(np.arange(len(transformed)) > self.dim // 4, 0, transformed)
        return np.real(np.fft.ifft(low_pass))

    def adaptive_learning_factors(self, personal_best, global_best, particle, inertia=0.5, cognitive=1.5, social=1.5):
        r1, r2 = np.random.rand(), np.random.rand()
        velocity = (
            inertia * particle["velocity"]
            + cognitive * r1 * (personal_best - particle["position"])
            + social * r2 * (global_best - particle["position"])
        )
        return velocity

    def stochastic_refinement(self, candidate, func, bounds):
        perturbation = np.random.normal(0, 0.01, size=self.dim)
        perturbed_candidate = candidate + perturbation
        res = minimize(func, perturbed_candidate, bounds=bounds, method='L-BFGS-B')
        refined = self.wavelet_transform(res.x)  # Enhanced local search with wavelet refinement
        return refined, res.fun

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        scaled_lb = np.zeros(self.dim)
        scaled_ub = np.ones(self.dim)

        pop_size = 20
        particles = [{
            "position": np.random.uniform(lb, ub, self.dim),
            "velocity": np.zeros(self.dim),
            "best_position": None,
            "best_fitness": float("inf")
        } for _ in range(pop_size)]

        global_best_position = None
        global_best_fitness = float("inf")

        fitness = np.array([func(p["position"]) for p in particles])
        for i, p in enumerate(particles):
            p["best_position"] = p["position"]
            p["best_fitness"] = fitness[i]
            if fitness[i] < global_best_fitness:
                global_best_position = p["position"]
                global_best_fitness = fitness[i]
        self.history.extend(fitness)
        budget_spent = len(particles)

        while budget_spent < self.budget:
            for i, p in enumerate(particles):
                if budget_spent >= self.budget:
                    break

                p["velocity"] = self.adaptive_learning_factors(p["best_position"], global_best_position, p)
                p["position"] = np.clip(p["position"] + p["velocity"], lb, ub)

                # Apply wavelet transformation to encourage periodicity
                p["position"] = self.wavelet_transform(p["position"])

                current_fitness = func(p["position"])
                budget_spent += 1
                self.history.append(current_fitness)

                if current_fitness < p["best_fitness"]:
                    p["best_position"] = p["position"]
                    p["best_fitness"] = current_fitness

                    if current_fitness < global_best_fitness:
                        global_best_position = p["position"]
                        global_best_fitness = current_fitness

            if budget_spent < self.budget:
                best_particle = min(particles, key=lambda x: x["best_fitness"])
                best_candidate, local_fitness = self.stochastic_refinement(best_particle["best_position"], func, list(zip(lb, ub)))

                if local_fitness < global_best_fitness:
                    global_best_position = best_candidate
                    global_best_fitness = local_fitness
                    budget_spent += 1
                    self.history.append(local_fitness)

        return global_best_position, global_best_fitness