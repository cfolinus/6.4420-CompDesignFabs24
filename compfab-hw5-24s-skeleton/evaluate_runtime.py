#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:54:29 2024

@author: cfolinus
"""
from pareto import pareto_front
import time
import numpy as np
from matplotlib import pyplot as plt

num_points = np.unique(np.logspace(0, 6, num=50, dtype=int))

num_trials = 1


# Initialize storage variables
evaluation_time_ms = np.zeros((len(num_points), num_trials))


for points_index in range(len(num_points)):
     
     temp_num_points = num_points[points_index]
     
     for trial_index in range(num_trials):
     
          # Generate random numbers
          points = np.random.uniform(size=(temp_num_points, 2))
          
          # Start timer
          tic = time.perf_counter()
          
          # Compute pareto front
          pareto_user = pareto_front(points)
          
          # Stop timer
          toc = time.perf_counter()
          net_time_ms = 1e3 * (toc - tic)
          
          # Store time
          evaluation_time_ms[points_index, trial_index] = net_time_ms


# Average across repeated trials at each number of points
evaluation_time_ms = np.mean(evaluation_time_ms, axis = 1)




# Plot results
fig, ax = plt.subplots(constrained_layout=True)
ax.scatter(num_points, evaluation_time_ms)

