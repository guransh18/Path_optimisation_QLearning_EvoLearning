import math
import numpy as np
import pygame
import os

# Import your existing objects
from car import Car
from NeuralNet import DQNAgent
from Evo import EvolutionSystem

# Import the heavy lifting from your existing NewTech file
from NewTech import (
    WIDTH, HEIGHT, S, SW, SH, CAR_RADIUS, RAY_ANGLES, RAY_QUERY_RADIUS,
    CELL_WALL, INITIAL_POINTS, CHECKPOINT_REWARD, POINT_DECAY_RATE,
    MODEL_SAVE_PATH, load_and_center_track, build_occupancy_grid,
    create_checkpoints_from_centerline, get_car_front, swept_capsule_query, 
    raycast_dda, get_state, find_first_checkpoint_ahead, draw_hud, 
    get_car_capsule, draw_capsule, BLACK, WHITE, RED, GREEN, BLUE
)

# ============================================================
# Presentation Config
# ============================================================
POPULATION_SIZE = 50     # Keep full population to run the genetic algorithm quietly
GHOST_COUNT     = 4      # How many runner-up cars to show
FIXED_DT        = 0.04   # 25 FPS RL physics logic
PRESENTATION_EPSILON = 0.01 # Tiny noise to prevent determinism death loops

# Curated list of tracks the model has mastered
# PROVEN_TRACKS = [
#     "track_dataset/train/hard/track_hard_0166.npy",
#     "track_dataset/train/hard/track_hard_0170.npy",
#     "track_dataset/train/hard/track_hard_0173.npy",
#     "track_dataset/train/hard/track_hard_0092.npy",
#     "track_dataset/train/hard/track_hard_0096.npy",
#     "track_dataset/train/hard/track_hard_0156.npy"
# ]

PROVEN_TRACKS = [
    "track_dataset/train/hard/track_hard_0136.npy"
]

def main():
    # 1. Load Initial Track
    current_track_idx = 0
    initial_track_path = PROVEN_TRACKS[current_track_idx]
    
    outerwall, innerwall, centerline = load_and_center_track(initial_track_path)
    current_track_name = os.path.basename(initial_track_path)
    all_walls = np.vstack([outerwall, innerwall])

    track_width = float(np.mean([np.linalg.norm(outerwall[i, :2] - innerwall[i, :2]) for i in range(10)]))
    _base_checkpoints = create_checkpoints_from_centerline(centerline, track_width=track_width, spacing_pixels=120)
    occ_grid = build_occupancy_grid(all_walls, _base_checkpoints)

    # 2. Find Start Pos
    start_x, start_y, start_angle = None, None, None
    for seg0 in centerline:
        cx, cy = (seg0[0] + seg0[2]) / 2, (seg0[1] + seg0[3]) / 2
        ang = math.atan2(seg0[3] - seg0[1], seg0[2] - seg0[0])
        test_pos = np.array([cx + math.cos(ang) * 10, cy + math.sin(ang) * 10])
        hit_type, _ = swept_capsule_query(test_pos, test_pos, ang, occ_grid)
        if hit_type != 'wall':
            start_x, start_y, start_angle = cx, cy, ang
            break

    first_cp = find_first_checkpoint_ahead(_base_checkpoints, start_x, start_y, start_angle)

    # 3. Setup Population
    cars = [Car(position=(start_x, start_y), angle=start_angle, screen_width=WIDTH, screen_height=HEIGHT) for _ in range(POPULATION_SIZE)]
    agents = [DQNAgent(state_size=7, action_size=8, hidden_size=16) for _ in range(POPULATION_SIZE)]
    evo_system = EvolutionSystem(population_size=POPULATION_SIZE)

    # Load pre-trained weights & Set Presentation Noise
    for idx, a in enumerate(agents):
        save_path = os.path.join(MODEL_SAVE_PATH, f"car_agent_{idx}.pth")
        if os.path.exists(save_path):
            a.load(save_path)
        a.epsilon = PRESENTATION_EPSILON  

    # 4. State Variables
    current_points         = [float(INITIAL_POINTS)] * POPULATION_SIZE
    checkpoints_passed     = [0] * POPULATION_SIZE
    current_checkpoint_idx = [first_cp] * POPULATION_SIZE
    laps_completed         = [0] * POPULATION_SIZE
    episode_fitnesses      = [0.0] * POPULATION_SIZE 
    alive_flags            = [True] * POPULATION_SIZE
    episode_count          = 1

    # 5. Pygame Setup
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AI Path Optimization - Presentation Mode")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont('Arial', max(12, int(round(24 * SH))))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        ray_debug_leader = None
        leader_idx = int(np.argmax(episode_fitnesses)) # Dynamically track the current best car

        # --- PHYSICS & AI UPDATE LOOP ---
        for i, car in enumerate(cars):
            if not alive_flags[i]:
                continue

            # AI Action (Forced Training=True to allow the 0.05 Epsilon)
            origin_pre  = get_car_front(car.position, car.angle)
            rays_pre    = [raycast_dda(origin_pre[0], origin_pre[1], car.angle + off, occ_grid) for off in RAY_ANGLES]
            current_state = get_state(car, rays_pre)
            
            action = agents[i].act(current_state, training=True) 
            keys   = agents[i].get_action_keys(action)

            prev_pos = car.position.copy()
            car.update(keys)

            step_reward = 0.0
            done = False

            # Collision & Checkpoints
            hit_type, hit_idx = swept_capsule_query(prev_pos, car.position, car.angle, occ_grid)

            if hit_type == 'wall':
                done = True
                step_reward *= 0.0
            elif hit_type == 'checkpoint' and hit_idx == current_checkpoint_idx[i]:
                current_points[i] += CHECKPOINT_REWARD
                checkpoints_passed[i] += 1
                current_checkpoint_idx[i] += 1
                step_reward += CHECKPOINT_REWARD
                
                if current_checkpoint_idx[i] >= len(_base_checkpoints):
                    laps_completed[i] += 1
                    current_checkpoint_idx[i] = 0
                    if laps_completed[i] >= 3:
                        done = True

            current_points[i] -= POINT_DECAY_RATE * FIXED_DT
            if not done: 
                step_reward += max(0.0, car.velocity * 0.2)

            if current_points[i] < 0:
                done = True

            episode_fitnesses[i] += step_reward

            if i == leader_idx:
                origin_local = get_car_front(car.position, car.angle)
                rays_local   = [raycast_dda(origin_local[0], origin_local[1], car.angle + off, occ_grid) for off in RAY_ANGLES]
                ray_debug_leader = (origin_local, car.angle, rays_local)

            if done:
                alive_flags[i] = False

        # --- RESET EPISODE ---
        if all(not f for f in alive_flags):
            episode_count += 1
            evo_system.evolve(agents, episode_fitnesses) 
            
            # --- Cycle to the next proven track ---
            current_track_idx = (current_track_idx + 1) % len(PROVEN_TRACKS)
            next_track_path = PROVEN_TRACKS[current_track_idx]
            
            outerwall, innerwall, centerline = load_and_center_track(next_track_path)
            all_walls = np.vstack([outerwall, innerwall])
            track_width = float(np.mean([np.linalg.norm(outerwall[i, :2] - innerwall[i, :2]) for i in range(10)]))
            _base_checkpoints = create_checkpoints_from_centerline(centerline, track_width=track_width, spacing_pixels=120)
            occ_grid = build_occupancy_grid(all_walls, _base_checkpoints)
            
            # --- Find new start position ---
            start_x, start_y, start_angle = None, None, None
            for seg0 in centerline:
                cx, cy = (seg0[0] + seg0[2]) / 2, (seg0[1] + seg0[3]) / 2
                ang = math.atan2(seg0[3] - seg0[1], seg0[2] - seg0[0])
                test_pos = np.array([cx + math.cos(ang) * 10, cy + math.sin(ang) * 10])
                hit_type, _ = swept_capsule_query(test_pos, test_pos, ang, occ_grid)
                if hit_type != 'wall':
                    start_x, start_y, start_angle = cx, cy, ang
                    break
            first_cp = find_first_checkpoint_ahead(_base_checkpoints, start_x, start_y, start_angle)

            # --- Respawn Population ---
            for i in range(POPULATION_SIZE):
                cars[i].position = np.array([start_x, start_y], dtype=float)
                cars[i].angle = start_angle
                cars[i].velocity = 0.0
                cars[i].steering = 0.0
                alive_flags[i] = True
                current_points[i] = float(INITIAL_POINTS)
                checkpoints_passed[i] = 0
                current_checkpoint_idx[i] = first_cp
                episode_fitnesses[i] = 0.0
                laps_completed[i] = 0
                agents[i].epsilon = PRESENTATION_EPSILON

        # --- DRAWING LOOP (THE GHOST SWARM) ---
        screen.fill(WHITE)

        # Draw Walls & Centerline
        for seg in outerwall: pygame.draw.line(screen, BLACK, (int(seg[0]), int(seg[1])), (int(seg[2]), int(seg[3])), 2)
        for seg in innerwall: pygame.draw.line(screen, BLACK, (int(seg[0]), int(seg[1])), (int(seg[2]), int(seg[3])), 2)
        for seg in centerline: pygame.draw.line(screen, RED, (int(seg[0]), int(seg[1])), (int(seg[2]), int(seg[3])), 1)

        # Sort indices by fitness to figure out who is winning right now
        sorted_indices = np.argsort(episode_fitnesses)[::-1]
        active_ghosts = 0

        # Draw Cars: Leader in Blue, Ghosts in Grey, skip the rest
        for rank, idx in enumerate(sorted_indices):
            if not alive_flags[idx]: continue
            
            if rank == 0:
                # The Leader
                cars[idx].draw(screen, BLUE)
            elif active_ghosts < GHOST_COUNT:
                # The Runner-Up Ghosts
                cars[idx].draw(screen, (200, 200, 200)) # Light Grey
                active_ghosts += 1

        # Draw Rays ONLY for the Leader
        if ray_debug_leader:
            origin, angle, dists = ray_debug_leader
            for off, dist in zip(RAY_ANGLES, dists):
                ang = angle + off
                ex, ey = origin[0] + math.cos(ang) * dist, origin[1] + math.sin(ang) * dist
                t = min(dist / RAY_QUERY_RADIUS, 1.0)
                col = (int(255 * (1 - t)), int(255 * t), 0)
                pygame.draw.line(screen, col, origin, (int(ex), int(ey)), 2)
                pygame.draw.circle(screen, (255, 80, 80), (int(ex), int(ey)), 3)

        # HUD
        draw_hud(screen, font, cars, alive_flags, current_points, checkpoints_passed, laps_completed, episode_count, episode_fitnesses, agents, len(_base_checkpoints))

        pygame.display.flip()
        clock.tick(60) # Presentation frame rate cap

    pygame.quit()

if __name__ == "__main__":
    main()