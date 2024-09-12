import numpy as np
import time
from joblib import Parallel, delayed
import os
import types
import warnings
import sys
import importlib
from utils import readTSPRandom
from gls.gls_run import solve_instance
import concurrent
import random


def solve_instance_parallel(i, opt_cost, instance, coord, time_limit, ite_max, perturbation_moves, heuristic_name):
    heuristic_module = importlib.import_module(f"{heuristic_name}")
    heuristic = importlib.reload(heuristic_module)
    return solve_instance(i, opt_cost, instance, coord, time_limit, ite_max, perturbation_moves, heuristic)


def solve_instance_parallel_TSP(i, opt_cost, instance, time_limit, ite_max, perturbation_moves, heuristic_name):
    heuristic_module = importlib.import_module(f"{heuristic_name}")
    heuristic = importlib.reload(heuristic_module)
    return solve_instance(i, opt_cost, instance, None, time_limit, ite_max, perturbation_moves, heuristic), i


class TSPGLS():
    def __init__(self) -> None:
        self.n_inst_eva = 64
        self.time_limit = 10 # maximum 10 seconds for each instance
        self.ite_max = 1000 # maximum number of local searchs in GLS for each instance
        self.perturbation_moves = 1 # movers of each edge in each perturbation
        path = os.path.dirname(os.path.abspath(__file__))
        self.instance_path = path+'/TrainingData/TSPAEL64.pkl' #,instances=None,instances_name=None,instances_scale=None
        self.debug_mode=False

        self.coords,self.instances,self.opt_costs = readTSPRandom.read_instance_all(self.instance_path)



    def tour_cost(self,instance, solution, problem_size):
        cost = 0
        for j in range(problem_size - 1):
            cost += np.linalg.norm(instance[int(solution[j])] - instance[int(solution[j + 1])])
        cost += np.linalg.norm(instance[int(solution[-1])] - instance[int(solution[0])])
        return cost

    def generate_neighborhood_matrix(self,instance):
        instance = np.array(instance)
        n = len(instance)
        neighborhood_matrix = np.zeros((n, n), dtype=int)

        for i in range(n):
            distances = np.linalg.norm(instance[i] - instance, axis=1)
            sorted_indices = np.argsort(distances)  # sort indices based on distances
            neighborhood_matrix[i] = sorted_indices

        return neighborhood_matrix
    

    def gls_instance(self, heuristic, seed):
        random.seed(seed)
        opt_costs, instances, coords = random.choice(list(zip(self.opt_costs, self.instances, self.coords)))
        return solve_instance(0, opt_costs,  
                                 instances, 
                                 coords,
                                 self.time_limit,
                                 self.ite_max,
                                 self.perturbation_moves,
                                 heuristic)

    def evaluateGLS(self,heuristic):

        gaps = np.zeros(self.n_inst_eva)

        for i in range(self.n_inst_eva):
            gap = solve_instance(i,self.opt_costs[i],  
                                 self.instances[i], 
                                 self.coords[i],
                                 self.time_limit,
                                 self.ite_max,
                                 self.perturbation_moves,
                                 heuristic)
            gaps[i] = gap

        return np.mean(gaps)
    
    def testGLS(self,heuristic_name,instance_dataset):
        self.debug_mode=False

        self.coords = instance_dataset['coordinate']
        optimal_tour = instance_dataset['optimal_tour']
        self.instances = instance_dataset['distance_matrix']
        self.opt_costs = instance_dataset['cost']

        #self.coords,self.instances,self.opt_costs = readTSPRandom.read_instance_all(self.instance_path)

        gaps = np.zeros(self.n_inst_eva)

        # Create a ProcessPoolExecutor with the number of available CPU cores
        with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
            # Submit tasks for parallel execution
            futures = [
                executor.submit(solve_instance_parallel, i, self.opt_costs[i], self.instances[i], 
                                self.coords[i], self.time_limit, self.ite_max, 
                                self.perturbation_moves, heuristic_name)
                for i in range(self.n_inst_eva)
            ]

            # Collect the results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                gaps[i] = future.result()

        return gaps

        # for i in range(self.n_inst_eva):
        #     gap = solve_instance(i,self.opt_costs[i],  
        #                          self.instances[i], 
        #                          self.coords[i],
        #                          self.time_limit,
        #                          self.ite_max,
        #                          self.perturbation_moves,
        #                          heuristic)
        #     gaps[i] = gap

        # return gaps
    

    # def evaluateGLS(self,heuristic):

    #     nins = 64    
    #     gaps = np.zeros(nins)

    #     print("Start evaluation ...")   

    #     inputs = [(x,self.opt_costs[x],  self.instances[x], self.coords[x],self.time_limit,self.ite_max,self.perturbation_moves) for x in range(nins)]
    #     #gaps = Parallel(n_jobs=nins)(delayed(solve_instance)(*input) for input in inputs)
    #     try:
    #             gaps = Parallel(n_jobs= 4, timeout = self.time_limit*1.1)(delayed(solve_instance)(*input) for input in inputs)
    #     except:
    #             print("### timeout or other error, return a large fitness value ###")
    #             return 1E10
    #     return np.mean(gaps)


    # def evaluate(self):
    #     try:        
    #         fitness = self.evaluateGLS()
    #         return fitness
    #     except Exception as e:
    #         print("Error:", str(e))  # Print the error message
    #         return None

    def evaluate(self, code_string):
        try:
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Create a new module object
                heuristic_module = types.ModuleType("heuristic_module")
                
                # Execute the code string in the new module's namespace
                exec(code_string, heuristic_module.__dict__)

                # Add the module to sys.modules so it can be imported
                sys.modules[heuristic_module.__name__] = heuristic_module

                #print(code_string)
                fitness = self.evaluateGLS(heuristic_module)

                return fitness
            
        except Exception as e:
            print("Error:", str(e))
            return None



