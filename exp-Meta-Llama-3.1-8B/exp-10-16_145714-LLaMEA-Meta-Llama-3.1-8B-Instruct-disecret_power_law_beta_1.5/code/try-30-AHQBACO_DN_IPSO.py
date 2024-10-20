import numpy as np
import random

class AHQBACO_DN_IPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory = np.random.uniform(-5, 5, (budget, dim))
        self.velocity = np.random.uniform(-1, 1, (budget, dim))
        self.best_solution = np.zeros(dim)
        self.best_fitness = np.inf
        self.employed_bees = np.random.uniform(-5, 5, (budget, dim))
        self.onlooker_bees = np.random.uniform(-5, 5, (budget, dim))
        self.dance_memory = np.zeros((budget, dim))
        self.neighborhood_size = 0.2 * budget  # dynamic neighborhood size
        self.learning_factor = 0.7298  # initial learning factor
        self.self_adaptive_velocity = np.zeros((budget, dim))  # self-adaptive velocity update rule

    def __call__(self, func):
        for i in range(self.budget):
            fitness = func(self.harmony_memory[i])
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = self.harmony_memory[i]
            # Harmony Search
            if random.random() < 0.5:
                self.harmony_memory[i] = self.harmony_memory[i] + np.random.uniform(-1, 1, self.dim)
                self.harmony_memory[i] = np.clip(self.harmony_memory[i], -5, 5)
            # Improved Quantum-Behaved Particle Swarm Optimization
            else:
                self.velocity[i] = self.learning_factor * self.velocity[i] + 1.49618 * (self.best_solution - self.harmony_memory[i]) + 1.49618 * (np.random.uniform(-1, 1, self.dim))
                self.self_adaptive_velocity[i] = 0.1 * np.random.uniform(-1, 1, self.dim) + 0.9 * self.velocity[i]
                self.harmony_memory[i] = self.harmony_memory[i] + self.self_adaptive_velocity[i]
                self.harmony_memory[i] = np.clip(self.harmony_memory[i], -5, 5)
                self.learning_factor = 0.5 + 0.5 * np.sin(10 * (i / self.budget))  # dynamic learning factor
            # Artificial Bee Colony Optimization
            if i < self.budget // 2:
                # Employed bees
                self.employed_bees[i] = self.harmony_memory[i] + np.random.uniform(-1, 1, self.dim)
                self.employed_bees[i] = np.clip(self.employed_bees[i], -5, 5)
                # Onlooker bees
                if random.random() < 0.5:
                    self.onlooker_bees[i] = self.employed_bees[i] + np.random.uniform(-self.neighborhood_size, self.neighborhood_size, self.dim)
                    self.onlooker_bees[i] = np.clip(self.onlooker_bees[i], -5, 5)
                else:
                    self.onlooker_bees[i] = self.dance_memory[i] + np.random.uniform(-1, 1, self.dim)
                    self.onlooker_bees[i] = np.clip(self.onlooker_bees[i], -5, 5)
                # Dance memory
                self.dance_memory[i] = self.onlooker_bees[i] if func(self.onlooker_bees[i]) < func(self.dance_memory[i]) else self.dance_memory[i]
            else:
                # Employed bees
                self.employed_bees[i] = self.onlooker_bees[i] + np.random.uniform(-1, 1, self.dim)
                self.employed_bees[i] = np.clip(self.employed_bees[i], -5, 5)
                # Onlooker bees
                if random.random() < 0.5:
                    self.onlooker_bees[i] = self.employed_bees[i] + np.random.uniform(-self.neighborhood_size, self.neighborhood_size, self.dim)
                    self.onlooker_bees[i] = np.clip(self.onlooker_bees[i], -5, 5)
                else:
                    self.onlooker_bees[i] = self.dance_memory[i] + np.random.uniform(-1, 1, self.dim)
                    self.onlooker_bees[i] = np.clip(self.onlooker_bees[i], -5, 5)
                # Dance memory
                self.dance_memory[i] = self.onlooker_bees[i] if func(self.onlooker_bees[i]) < func(self.dance_memory[i]) else self.dance_memory[i]
            self.neighborhood_size = max(0.1 * budget, self.neighborhood_size * 0.9)  # adjust neighborhood size based on fitness of employed bees
        return self.best_solution, self.best_fitness

# Example usage:
def sphere(x):
    return np.sum(x**2)

budget = 1000
dim = 10
ahqbaco_dn_ipso = AHQBACO_DN_IPSO(budget, dim)
best_solution, best_fitness = ahqbaco_dn_ipso(sphere)
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)