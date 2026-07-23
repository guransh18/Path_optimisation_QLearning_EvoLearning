# 🏎️ Hybrid DQN-EA Autonomous Navigation Simulator

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Dynamic%20Deep%20Learning-EE4C2C?logo=pytorch)
![PyGame](https://img.shields.io/badge/PyGame-2D%20Rendering-green)
![Status](https://img.shields.io/badge/Status-MSc%20Thesis%20Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)

> An advanced, highly optimized 2D autonomous driving simulation combining the reactive gradient descent of a Deep Q-Network (DQN) with the global optimization of an Evolutionary Algorithm (EA). 

## 📖 Project Abstract
Standard autonomous reinforcement learning (RL) agents often suffer from two major bottlenecks: susceptibility to catastrophic local minima (e.g., "idle lock") and extreme computational overhead during physics calculations. 

This project solves these issues via a **two-tiered Hybrid Architecture**. A PyTorch-based Dueling DQN acts as the intra-episode *Micro-Learner* for frame-by-frame steering, while a NumPy-based Evolutionary Algorithm acts as the inter-episode *Macro-Learner*, applying Tournament Selection, N-Point Crossover, and Gaussian Mutation to rescue stagnant agents. Furthermore, standard $O(N \times M)$ physics calculations were entirely bypassed using an $O(1)$ Occupancy Grid and Digital Differential Analyzer (DDA) raycasting, allowing for massive population scaling on consumer hardware.

---

## ✨ Core Innovations & Features

### 🧠 1. The "Train Fast, Test Slow" Paradigm
The execution engine strictly decouples machine learning logic from the rendering pipeline. 
* **Training Mode (`DEMO_MODE = False`):** Operates completely headlessly (`RENDER_EVERY = 999999`). PyGame's pixel-pushing is disabled, allowing the CPU/GPU to process matrices and tensor mathematics synchronously at maximum hardware limits.
* **Presentation Mode (`DEMO_MODE = True`):** Clamps the physics engine to a cinematic 60 FPS (`FIXED_DT = 0.016`), forces 100% exploitation (`epsilon = 0.0`), and loads the optimized `best_agent.pth` weights for silky-smooth visual inference.

### ⚡ 2. O(1) Computational Physics
* **Occupancy Grids:** Continuous vector geometries are rasterized into a discrete 2D matrix (`OCC_CELL_SIZE = 4`), reducing collision detection to a near-instantaneous $O(1)$ spatial hash lookup.
* **DDA Raycasting:** The agent's 5-dimensional LiDAR vision uses the Digital Differential Analyzer (DDA) algorithm to step through the grid cells rather than calculating floating-point trigonometry, reducing computational load by >90%.

### 🧬 3. Flat-Memory Genetic Bridging
To prevent memory allocation lag when passing data between the GPU and CPU, the system features a custom bridge that utilizes PyTorch's `state_dict()` to instantly flatten multi-dimensional tensors into contiguous 1D NumPy arrays for genetic manipulation, seamlessly reshaping them back for network reinjection[cite: 2, 4].

### 🌍 4. Procedural Track Generation (CSG)
To prevent neural network overfitting to static tracks, the simulation includes an offline Constructive Solid Geometry (CSG) pipeline utilizing the `shapely` library. It autonomously extrudes valid, closed-loop track topologies, segregating them into an 80/20 train/test split across easy, medium, and hard curriculum difficulties.

---

## 🏗️ System Architecture

* **Micro-Learner (PyTorch Dueling DQN):** 7-dimensional continuous state space (velocity, steering, 5 normalized rays) mapping to 8 discrete driving combinations. The action space maps index `0` to acceleration to systematically prevent "idle lock".
* **Macro-Learner (Evolutionary System):** Executes synchronously upon total population termination. Enforces elitism, tournament selection ($k = \max(5, \text{pop} \times 0.07)$), and multi-point genomic crossover.
* **Spatial Evaluator (Reward System):** Utilizes mathematically generated perpendicular gates and Continuous Collision Detection (CCD) vector intersections to track fitness with absolute precision, immune to high-velocity physics tunneling.

---
## 📊 Evaluation & Metrics
*During headless training, the system dynamically generates a training_metrics.csv dashboard tracking Episode Iterations, Best Fitness, Average Fitness, and Epsilon Decay rates. The strictly positive, bounded reward function guarantees clean gradients while effectively penalizing catastrophic collisions via multiplicative crushing (reward *= 0.0).

---
## 📂 Repository Structure
```text
📦 PATH_OPTIMIZATION
 ┣ 📂 Car_Agent_Data/       # Serialized PyTorch weight dictionaries (.pth)
 ┣ 📂 track_dataset/        # Procedurally generated CSG numpy tracks (Train/Test)
 ┣ 📜 NewTech.py            # Main execution engine, Physics, & PyGame loop
 ┣ 📜 NeuralNet.py          # PyTorch Dueling DQN & Experience Replay buffer
 ┣ 📜 Evo.py                # Evolutionary Algorithm macro-learner
 ┣ 📜 Checkpoint.py         # Dynamic spatial gating & CCD reward logic
 ┣ 📜 car.py                # 2D Kinematics and local-to-global transformations
 ┣ 📜 TrackGenerator.py     # Offline procedural generation pipeline
 ┣ 📜 Manual_Track.py       # PyQt5 GUI for manual track drawing/debugging
 ┗ 📜 track_viewer.py       # Matplotlib diagnostic utility for track datasets
