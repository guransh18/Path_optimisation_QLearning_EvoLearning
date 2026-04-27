"""
NewTech.py  –  Responsive multi-car DQN training visualiser.

All track geometry, car physics, UI layout and font sizes derive from the
two constants at the top of this file (WIDTH / HEIGHT).  Change them and
everything scales automatically.
"""

import math
import numpy as np
import pygame
from collections import defaultdict
from car import Car
from NeuralNet import DQNAgent
from Evo import EvolutionSystem
import csv
import os
import random
import glob
import copy
from Checkpoint import create_checkpoints_from_centerline, update_checkpoints, draw_checkpoints,reset_checkpoints,Checkpoint

# ============================================================
#  >>>  CHANGE THESE TWO VALUES TO RESIZE EVERYTHING  <<<
# ============================================================
WIDTH  = 1280
HEIGHT = 720
# ============================================================

# ---------------------------------------------------------------------------
# Derived scale helpers
# ---------------------------------------------------------------------------
_BASE_W, _BASE_H = 1280, 720

def _s() -> float:
    """Uniform pixel-scale relative to the 1280×720 baseline."""
    return math.hypot(WIDTH, HEIGHT) / math.hypot(_BASE_W, _BASE_H)


def _sw() -> float:
    """Horizontal scale factor."""
    return WIDTH / _BASE_W


def _sh() -> float:
    """Vertical scale factor."""
    return HEIGHT / _BASE_H


S  = _s()   # uniform scale (computed once at import time)
SW = _sw()
SH = _sh()


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
BLACK  = (0,   0,   0)
WHITE  = (255, 255, 255)
RED    = (255, 0,   0)
BLUE   = (0,   0,   255)
GREEN  = (0,   255, 0)


# ---------------------------------------------------------------------------
# Geometry / physics constants  (all proportional to S / SW / SH)
# ---------------------------------------------------------------------------

# Car capsule for collision detection
CAR_LENGTH      = max(6,  int(round(30 * S)))
CAR_RADIUS      = max(3,  int(round(10 * S)))   # half-width for capsule

# Spatial grid cell size scales with the average track dimension
# CELL_SIZE       = max(40, int(round(100 * S)))

# Raycasting
RAY_QUERY_RADIUS = max(100, int(round(300 * S)))
RAY_ANGLES = [
    0.0,
    math.radians(30),
    math.radians(-30),
    math.radians(60),
    math.radians(-60)
]

# # Car query radius for nearby segment lookup
# CAR_QUERY_RADIUS = max(15, int(round(30 * S)))

# Occupancy grid
OCC_CELL_SIZE = 4  # pixels per grid cell (lower = more precise, more RAM)
CELL_WALL       = 1
CELL_CHECKPOINT = 2
# Visual flags
VISUALIZE_DEBUG = True

# Population

CAR_COLOR_ALIVE = BLUE
CAR_COLOR_DEAD  = RED

# Checkpoints
NUM_CHECKPOINTS = 30

# Point / reward system
INITIAL_POINTS     = 100
CHECKPOINT_REWARD  = 100
POINT_DECAY_RATE   = 25     # per second

# ============================================================
# Mode Toggles
# ============================================================
DEMO_MODE       = True  # Set to True for Defense Day (Shows 1 perfect car)

DEMO_TRACK_PATH = "track_dataset/train/hard/track_hard_0156.npy"

if DEMO_MODE:
    TRAINING_MODE   = False
    POPULATION_SIZE = 1
    RENDER_EVERY    = 1
    FIXED_DT        = 0.04  # 60 FPS silky smooth physics for presentation
else:
    TRAINING_MODE   = True
    POPULATION_SIZE = 50
    RENDER_EVERY    = 999999 # Effectively Headless for max speed
    FIXED_DT        = 0.04   # 25 FPS RL physics logic

# Evolution Constants
MODEL_SAVE_PATH     = "Car_Agent_Data"
CSV_LOG_PATH        = "training_metrics.csv"
STATE_SIZE          = 7     # velocity, steering, 5 ray distances
ACTION_SIZE         = 8     # W, A, S, D, and combinations + Idle
MODEL_SAVE_PATH     = "Car_Agent_Data"
TARGET_UPDATE_FREQ  = 100
CONSOLE_METRICS     = True
PRINT_EVERY_EPISODE = 1

# ---------------------------------------------------------------------------
# Font sizes (proportional to window height, clamped to a readable minimum)
# ---------------------------------------------------------------------------
FONT_SIZE_NORMAL = max(12, int(round(24 * SH)))
FONT_SIZE_SMALL  = max(10, int(round(18 * SH)))

# Track Rotation
current_track_path = None

# ===========================================================================
#  Spatial grid helpers
# ===========================================================================

# def build_spatial_grid(segments, cell_size=CELL_SIZE):
#     grid = defaultdict(list)
#     for i, (x1, y1, x2, y2) in enumerate(segments):
#         cx0 = int(min(x1, x2) // cell_size)
#         cx1 = int(max(x1, x2) // cell_size)
#         cy0 = int(min(y1, y2) // cell_size)
#         cy1 = int(max(y1, y2) // cell_size)
#         for cx in range(cx0, cx1 + 1):
#             for cy in range(cy0, cy1 + 1):
#                 grid[(cx, cy)].append(i)
#     return grid


# def query_spatial_grid(grid, segments, x, y, radius, cell_size=CELL_SIZE):
#     cx = int(x // cell_size)
#     cy = int(y // cell_size)
#     r  = int(math.ceil(radius / cell_size))
#     idx = set()
#     for dx in range(-r, r + 1):
#         for dy in range(-r, r + 1):
#             cell = (cx + dx, cy + dy)
#             if cell in grid:
#                 idx.update(grid[cell])
#     if not idx:
#         return np.empty((0, 4))
#     return segments[list(idx)]


# # ===========================================================================
# #  Geometry helpers
# # ===========================================================================

# def point_segment_dist_sq(px, py, x1, y1, x2, y2):
#     vx, vy = x2 - x1, y2 - y1
#     wx, wy = px - x1, py - y1
#     c1 = vx * wx + vy * wy
#     if c1 <= 0:
#         return (px - x1) ** 2 + (py - y1) ** 2
#     c2 = vx * vx + vy * vy
#     if c2 <= c1:
#         return (px - x2) ** 2 + (py - y2) ** 2
#     b = c1 / c2
#     return (px - x1 - b * vx) ** 2 + (py - y1 - b * vy) ** 2


# def segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
#     def orient(ax, ay, bx, by, cx, cy):
#         return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
#     o1 = orient(x1, y1, x2, y2, x3, y3)
#     o2 = orient(x1, y1, x2, y2, x4, y4)
#     o3 = orient(x3, y3, x4, y4, x1, y1)
#     o4 = orient(x3, y3, x4, y4, x2, y2)
#     return (o1 * o2 < 0) and (o3 * o4 < 0)


# def segment_segment_dist_sq(x1, y1, x2, y2, x3, y3, x4, y4):
#     if segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
#         return 0.0
#     return min(
#         point_segment_dist_sq(x1, y1, x3, y3, x4, y4),
#         point_segment_dist_sq(x2, y2, x3, y3, x4, y4),
#         point_segment_dist_sq(x3, y3, x1, y1, x2, y2),
#         point_segment_dist_sq(x4, y4, x1, y1, x2, y2),
#     )



def get_random_track(mode="train", difficulty="medium", base_dir="track_dataset"):
    """
    Pulls a random track from the correct ML split and difficulty bin.
    """
    search_path = os.path.join(base_dir, mode, difficulty, "*.npy")
    available_tracks = glob.glob(search_path)
    
    if not available_tracks:
        # Fallback if the folder is empty/missing
        print(f"[ERROR] No tracks found in {search_path}. Using static fallback.")
        return "Track1_combined_segments.npy"
        
    return random.choice(available_tracks)

def get_car_front(position, angle, length=CAR_LENGTH):
    x, y = position
    return x + math.cos(angle) * (length / 2), y + math.sin(angle) * (length / 2)


def get_car_capsule(position, angle, length=CAR_LENGTH):
    x, y = position
    dx, dy = math.cos(angle), math.sin(angle)
    half = length / 2
    return x - dx * half, y - dy * half, x + dx * half, y + dy * half

def find_first_checkpoint_ahead(checkpoints, start_x, start_y, start_angle):
    """Finds the first checkpoint that is physically in front of the car's spawn point."""
    forward = np.array([math.cos(start_angle), math.sin(start_angle)])
    for idx, cp in enumerate(checkpoints):
        to_cp = cp.center - np.array([start_x, start_y])
        if np.dot(to_cp, forward) > 0:  # checkpoint is in front of car
            return idx
    return 0  # fallback

def track_is_valid(start_x, start_y, start_angle, occ_grid):
    """Simulates 3 frames of forward movement using the swept capsule to ensure safety."""
    prev_pos = np.array([start_x, start_y], dtype=float)
    
    for _ in range(3):
        # Simulate moving forward by 5 pixels per frame
        curr_pos = prev_pos + np.array([math.cos(start_angle), math.sin(start_angle)]) * 5.0
        
        # Use the new unified engine to check for wall collisions
        hit_type, _ = swept_capsule_query(prev_pos, curr_pos, start_angle, occ_grid)
        
        if hit_type == 'wall':
            return False
            
        prev_pos = curr_pos
        
    return True


#Occupancy grid
# def build_occupancy_grid(segments, cell_size=OCC_CELL_SIZE):
#     cols = math.ceil(WIDTH  / cell_size) + 1
#     rows = math.ceil(HEIGHT / cell_size) + 1
#     grid = np.zeros((rows, cols), dtype=np.uint8)
#     for x1, y1, x2, y2 in segments:
#         steps = int(math.hypot(x2 - x1, y2 - y1) / cell_size * 2) + 2
#         for k in range(steps + 1):
#             t  = k / steps
#             gx = int((x1 + t * (x2 - x1)) / cell_size)
#             gy = int((y1 + t * (y2 - y1)) / cell_size)
#             if 0 <= gy < rows and 0 <= gx < cols:
#                 grid[gy, gx] = 1
#     return grid

def build_occupancy_grid(segments, checkpoint_list=None, cell_size=OCC_CELL_SIZE):
    cols = math.ceil(WIDTH  / cell_size) + 1
    rows = math.ceil(HEIGHT / cell_size) + 1
    
    # Use uint16 to safely support tracks with up to 65,000 checkpoints
    grid = np.zeros((rows, cols), dtype=np.uint16) 

    # 1. Paint walls
    for x1, y1, x2, y2 in segments:
        steps = int(math.hypot(x2 - x1, y2 - y1) / cell_size * 2) + 2
        for k in range(steps + 1):
            t  = k / steps
            gx = int((x1 + t * (x2 - x1)) / cell_size)
            gy = int((y1 + t * (y2 - y1)) / cell_size)
            if 0 <= gy < rows and 0 <= gx < cols:
                grid[gy, gx] = CELL_WALL

    # 2. Paint checkpoints (only if not overwriting a wall)
    if checkpoint_list:
        for cp_idx, cp in enumerate(checkpoint_list):
            x1, y1 = cp.point1
            x2, y2 = cp.point2
            steps = int(math.hypot(x2-x1, y2-y1) / cell_size * 2) + 2
            for k in range(steps + 1):
                t = k / steps
                gx = int((x1 + t*(x2-x1)) / cell_size)
                gy = int((y1 + t*(y2-y1)) / cell_size)
                if 0 <= gy < rows and 0 <= gx < cols:
                    if grid[gy, gx] != CELL_WALL:
                        grid[gy, gx] = CELL_CHECKPOINT + cp_idx  # Encode checkpoint ID into grid
    return grid

def swept_capsule_query(prev_pos, curr_pos, angle, occ_grid,
                        cell_size=OCC_CELL_SIZE,
                        car_half_width=CAR_RADIUS,
                        sweep_steps=6):
    """
    Returns: ('wall', None) | ('checkpoint', cp_index) | (None, None)
    Tests intermediate positions to prevent frame-skipping / tunneling.
    """
    rows, cols = occ_grid.shape

    for step in range(sweep_steps + 1):
        t = step / sweep_steps
        px = prev_pos[0] + t * (curr_pos[0] - prev_pos[0])
        py = prev_pos[1] + t * (curr_pos[1] - prev_pos[1])

        # Capsule width: sample perpendicular offsets
        cos_a, sin_a = math.cos(angle + math.pi/2), math.sin(angle + math.pi/2)
        for w in np.linspace(-car_half_width, car_half_width, 5):
            sx = px + cos_a * w
            sy = py + sin_a * w
            gx = int(sx / cell_size)
            gy = int(sy / cell_size)
            if not (0 <= gy < rows and 0 <= gx < cols):
                continue
            
            cell = occ_grid[gy, gx]
            if cell == CELL_WALL:
                return 'wall', None
            if cell >= CELL_CHECKPOINT:
                return 'checkpoint', int(cell - CELL_CHECKPOINT)

    return None, None

# def car_collides_occ(position, angle, occ_grid, cell_size=OCC_CELL_SIZE):
#     ax, ay, bx, by = get_car_capsule(position, angle)
    
#     # Calculate the perpendicular vector for the car's width
#     # Using CAR_RADIUS (half-width) to push the lines left and right
#     nx, ny = -math.sin(angle) * CAR_RADIUS, math.cos(angle) * CAR_RADIUS
    
#     # We will check 3 lines: The center spine, the left edge, and the right edge
#     lines_to_check = [
#         (ax, ay, bx, by),                                     # Center
#         (ax + nx, ay + ny, bx + nx, by + ny),                 # Left Edge
#         (ax - nx, ay - ny, bx - nx, by - ny)                  # Right Edge
#     ]
    
#     for x1, y1, x2, y2 in lines_to_check:
#         steps = max(int(math.hypot(x2 - x1, y2 - y1) / cell_size * 2), 4)
#         for k in range(steps + 1):
#             t  = k / steps
#             gx = int((x1 + t * (x2 - x1)) / cell_size)
#             gy = int((y1 + t * (y2 - y1)) / cell_size)
            
#             # If any point on any of the 3 lines hits a wall, the car is dead
#             if (0 <= gy < occ_grid.shape[0] and
#                     0 <= gx < occ_grid.shape[1] and
#                     occ_grid[gy, gx] == 1):
#                 return True
                
#     return False


def raycast_dda(ray_x, ray_y, ray_angle, occ_grid,
                cell_size=OCC_CELL_SIZE, max_dist=RAY_QUERY_RADIUS):
    dx, dy   = math.cos(ray_angle), math.sin(ray_angle)
    gx, gy   = int(ray_x / cell_size), int(ray_y / cell_size)
    step_x   = 1 if dx >= 0 else -1
    step_y   = 1 if dy >= 0 else -1
    t_delta_x = abs(cell_size / dx) if dx != 0 else float('inf')
    t_delta_y = abs(cell_size / dy) if dy != 0 else float('inf')
    t_max_x   = ((gx + (step_x > 0)) * cell_size - ray_x) / dx if dx != 0 else float('inf')
    t_max_y   = ((gy + (step_y > 0)) * cell_size - ray_y) / dy if dy != 0 else float('inf')
    rows, cols = occ_grid.shape
    t = 0.0
    while t < max_dist:
        if t_max_x < t_max_y:
            t   = t_max_x
            gx += step_x
            t_max_x += t_delta_x
        else:
            t   = t_max_y
            gy += step_y
            t_max_y += t_delta_y
        if not (0 <= gy < rows and 0 <= gx < cols):
            return max_dist
        if occ_grid[gy, gx] == CELL_WALL:
            return min(t, max_dist)
    return max_dist


# ===========================================================================
#  Track loading  –  segments are centered on (WIDTH/2, HEIGHT/2)
# ===========================================================================

def load_and_center_track(path: str):
    """
    Load Track1_combined_segments.npy, scale it so its bounding box fills
    ~80 % of the window, and center it exactly on (WIDTH/2, HEIGHT/2).

    Returns (outerwall, innerwall, centerline) as float32 numpy arrays of
    shape (N, 4) where each row is (x1, y1, x2, y2).
    """
    segs = np.load(path, allow_pickle=False).astype(np.float64)
    segs = segs.reshape(-1, 4)

    # Compute bounding box of all endpoint coordinates
    all_x = np.concatenate([segs[:, 0], segs[:, 2]])
    all_y = np.concatenate([segs[:, 1], segs[:, 3]])
    min_x, max_x = all_x.min(), all_x.max()
    min_y, max_y = all_y.min(), all_y.max()
    track_w = max(1.0, max_x - min_x)
    track_h = max(1.0, max_y - min_y)

    # Scale to fill 80 % of the window, preserving aspect ratio
    margin = 0.80
    scale = min(WIDTH * margin / track_w, HEIGHT * margin / track_h)

    cx_src = (min_x + max_x) / 2
    cy_src = (min_y + max_y) / 2

    def _transform(arr):
        out = arr.copy()
        out[:, 0] = (arr[:, 0] - cx_src) * scale + WIDTH  / 2
        out[:, 1] = (arr[:, 1] - cy_src) * scale + HEIGHT / 2
        out[:, 2] = (arr[:, 2] - cx_src) * scale + WIDTH  / 2
        out[:, 3] = (arr[:, 3] - cy_src) * scale + HEIGHT / 2
        return out.astype(np.float32)

    segs_t  = _transform(segs)
    n       = len(segs_t)
    half    = n // 2

    outerwall  = segs_t[:half]
    innerwall  = segs_t[half:]
    centerline = 0.5 * (outerwall + innerwall[:len(outerwall)])

    return outerwall, innerwall, centerline


# ===========================================================================
#  Checkpoint factory  (centerline-segment based)
# ===========================================================================



# ===========================================================================
#  Drawing helpers
# ===========================================================================

def draw_capsule(surface, ax, ay, bx, by, radius, color=GREEN):
    pygame.draw.line(surface, color, (int(ax), int(ay)), (int(bx), int(by)), 2)
    pygame.draw.circle(surface, color, (int(ax), int(ay)), radius, 2)
    pygame.draw.circle(surface, color, (int(bx), int(by)), radius, 2)
    vx, vy = bx - ax, by - ay
    length = math.hypot(vx, vy)
    if length == 0:
        return
    nx, ny = -vy / length, vx / length
    pygame.draw.line(surface, color,
                     (int(ax + nx * radius), int(ay + ny * radius)),
                     (int(bx + nx * radius), int(by + ny * radius)), 2)
    pygame.draw.line(surface, color,
                     (int(ax - nx * radius), int(ay - ny * radius)),
                     (int(bx - nx * radius), int(by - ny * radius)), 2)


def draw_hud(surface, font, cars, alive_flags, current_points,
             checkpoints_passed, laps_completed, episode_count,
             episode_rewards, agents, total_checkpoints):
    """Draw all HUD text, positioned as fractions of HEIGHT."""
    line_h = FONT_SIZE_NORMAL + 4    # line height in pixels
    x      = max(6, int(10 * SW))   # left margin

    best_idx  = int(np.argmax(current_points))
    worst_idx = int(np.argmin(current_points))
    best_car  = cars[best_idx]
    car_x, car_y = best_car.position
    angle_deg = math.degrees(best_car.angle) % 360

    lines = [
        (f"Best Pos: ({car_x:.1f}, {car_y:.1f})",            BLACK),
        (f"Best Angle: {angle_deg:.1f}°",                     BLACK),
        (f"Best Velocity: {best_car.velocity:.2f}",           BLACK),
        ("",                                                   BLACK),
        (f"Best Points:  {current_points[best_idx]:.1f}",
             (0, 150, 0) if current_points[best_idx] > 50
             else (255, 165, 0) if current_points[best_idx] > 0
             else RED),
        (f"Worst Points: {current_points[worst_idx]:.1f}",    BLACK),
        (f"Best Checkpts: {checkpoints_passed[best_idx]}/{total_checkpoints}", BLACK), # <--- UPDATED
        (f"Best Laps: {laps_completed[best_idx]}",            BLACK),
    ]

    if TRAINING_MODE:
        avg_eps = float(np.mean([a.epsilon for a in agents])) if agents else 0.0
        avg_rew = float(np.mean(episode_rewards))             if episode_rewards else 0.0
        lines += [
            ("",                                               BLACK),
            (f"Episode: {episode_count}",                     BLACK),
            (f"Avg ε: {avg_eps:.3f}",                        BLACK),
            (f"Avg Reward: {avg_rew:.1f}",                   BLACK),
            ("MODE: TRAINING",                                RED),
        ]
    else:
        lines += [
            ("",                                               BLACK),
            ("MODE: MANUAL",                                  (0, 150, 0)),
        ]

    for i, (text, color) in enumerate(lines):
        if text:
            surf = font.render(text, True, color)
            surface.blit(surf, (x, int(10 * SH) + i * line_h))


# ===========================================================================
#  Metrics / console
# ===========================================================================

def print_enhanced_metrics(episode_idx, fitnesses, checkpoints_list, 
                           laps_list, agents, episode_time, done_reasons, total_checkpoints, track_name="Unknown"):
    """Prints a detailed dashboard to the console for headless monitoring."""
    fitnesses = np.array(fitnesses)
    best_idx = np.argmax(fitnesses)
    worst_idx = np.argmin(fitnesses)
    
    # Population Stats
    avg_fit = np.mean(fitnesses)
    top_5_avg = np.mean(np.sort(fitnesses)[-5:]) if len(fitnesses) >= 5 else avg_fit
    avg_eps = np.mean([a.epsilon for a in agents])
    avg_time = np.mean(episode_time)
    
    # Cause of Death tracking
    collisions = done_reasons.count("collision")
    depleted = done_reasons.count("points_depleted")
    
    print("\n" + "="*60)
    print(f" GENERATION / EPISODE {episode_idx} ".center(60, "="))
    print("="*60)
    print(f" [Current Track]:   {track_name}") # <--- ADDED TRACK NAME HERE
    print(f" [Time Elapsed]:    {avg_time:.1f} seconds (Sim Time)")
    print(f" [Exploration e]:   {avg_eps:.4f} (Avg)")
    print("-" * 60)
    print(f" [Avg Fitness]:     {avg_fit:.1f}")
    print(f" [Top 5 Elites]:    {top_5_avg:.1f} (Avg Fitness)")
    print("-" * 60)
    print(f" >> CAUSE OF DEATH <<")
    print(f"    Hit Wall:       {collisions} cars")
    print(f"    Ran Out of Time:{depleted} cars")
    print("-" * 60)
    print(f" >> BEST AGENT (#{best_idx}) <<")
    print(f"    Fitness:        {fitnesses[best_idx]:.1f}")
    print(f"    Checkpoints:    {checkpoints_list[best_idx]}/{total_checkpoints}")
    print(f"    Laps:           {laps_list[best_idx]}")
    print("-" * 60)
    
    # Print the grid of all agents
    print(" >> ALL AGENT FITNESS SCORES <<")
    grid_str = "    "
    for i, fit in enumerate(fitnesses):
        grid_str += f"[{i:02d}: {fit:5.0f}] "
        if (i + 1) % 10 == 0 and i != len(fitnesses) - 1:
            grid_str += "\n    "
    print(grid_str)
    print("="*60 + "\n")


# ===========================================================================
#  State helper
# ===========================================================================

def get_state(car, ray_distances):
    v_norm = car.velocity / car.max_velocity
    s_norm = (car.steering /
              (car.steering_speed * car.turning_circle)
              if (car.steering_speed * car.turning_circle) != 0 else 0.0)
    rays_n = [d / RAY_QUERY_RADIUS for d in ray_distances]
    return [v_norm, s_norm] + rays_n


# ===========================================================================
#  Main
# ===========================================================================

def main():
    # ------------------------------------------------------------------
    # Load track
    # ------------------------------------------------------------------
    # track_path = "Track1_combined_segments.npy"
    _mode = "test" if DEMO_MODE else "train"
    _diff = "hard" if DEMO_MODE else "easy"
    
    # initial_track_path = get_random_track(mode=_mode, difficulty=_diff)

    if DEMO_MODE and DEMO_TRACK_PATH:
        initial_track_path = DEMO_TRACK_PATH
    else:
        initial_track_path = get_random_track(mode=_mode, difficulty=_diff)
    outerwall, innerwall, centerline = load_and_center_track(initial_track_path)
    current_track_name = os.path.basename(initial_track_path)

    # Estimate track width for checkpoints
    n_samp = min(len(outerwall), len(innerwall))
    track_width = float(np.mean([
        np.linalg.norm(outerwall[i, :2] - innerwall[i, :2])
        for i in range(n_samp)
    ]))

    # Wall collection for collision / raycasting
    all_walls = np.vstack([outerwall, innerwall])

    _base_checkpoints = create_checkpoints_from_centerline(
        centerline, track_width=track_width, spacing_pixels=120
    )

    # wall_grid = build_spatial_grid(all_walls)
    occ_grid = build_occupancy_grid(all_walls,_base_checkpoints)
    # Checkpoints — one independent copy per car

    # car_checkpoints = [copy.deepcopy(_base_checkpoints) for _ in range(POPULATION_SIZE)]

    # ------------------------------------------------------------------
    # Start position (midpoint of first centerline segment)
    # ------------------------------------------------------------------
    seg0       = centerline[0]
    start_x    = (seg0[0] + seg0[2]) / 2
    start_y    = (seg0[1] + seg0[3]) / 2
    start_angle = math.atan2(seg0[3] - seg0[1], seg0[2] - seg0[0])
    # --- ADD THIS: Find the correct starting checkpoint ---
    first_cp = find_first_checkpoint_ahead(_base_checkpoints, start_x, start_y, start_angle)
    print(f"[STARTUP] Starting at checkpoint index {first_cp} / {len(_base_checkpoints)}")  # ← add this

    # ------------------------------------------------------------------
    # Cars  – pass window size so physics scale correctly
    # ------------------------------------------------------------------
    cars = [
        Car(position=(start_x, start_y), angle=start_angle,
            screen_width=WIDTH, screen_height=HEIGHT)
        for _ in range(POPULATION_SIZE)
    ]

    # ------------------------------------------------------------------
    # DQN Agents
    # ------------------------------------------------------------------
    agents = [
        DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE,
                 hidden_size=16, learning_rate=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 memory_size=10_000, batch_size=64)
        for _ in range(POPULATION_SIZE)
    ]

    # Ensure the save directory exists
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    # Attempt to load previous weights for each agent
    # for idx, a in enumerate(agents):
    #     save_path = os.path.join(MODEL_SAVE_PATH, f"car_agent_{idx}.pth")
    #     if os.path.exists(save_path):
    #         a.load(save_path)
    #         # Optional: If you want to force a specific exploration rate after loading, 
    #         # you can overwrite it here. Otherwise, it uses the saved epsilon.
    #         # a.epsilon = max(a.epsilon, 0.1) 
    #     else:
    #         print(f"No saved data found for agent {idx}. Starting fresh.")

    if DEMO_MODE:
        # Load the single best agent for the presentation
        best_path = os.path.join(MODEL_SAVE_PATH, "best_agent.pth")
        if os.path.exists(best_path):
            agents[0].load(best_path)
            agents[0].epsilon = 0.01  # Force 100% exploitation (no random turns)
            print(">>> DEMO MODE ACTIVE: Best Agent Loaded <<<")
        else:
            print("ERROR: best_agent.pth not found. Run training first!")
    else:
        # Load previous weights if resuming headless training
        for idx, a in enumerate(agents):
            save_path = os.path.join(MODEL_SAVE_PATH, f"car_agent_{idx}.pth")
            if os.path.exists(save_path):
                a.load(save_path)
        for a in agents:
            a.epsilon = max(a.epsilon, 0.30)
            a.epsilon_decay = 0.9995
            a.epsilon_min   = 0.01
        print(f"[RESUME] Epsilon forced to 0.30, decay=0.9995")

    # ------------------------------------------------------------------
    # Episode state
    # ------------------------------------------------------------------
    current_points         = [float(INITIAL_POINTS)] * POPULATION_SIZE
    checkpoints_passed     = [0]  * POPULATION_SIZE
    current_checkpoint_idx = [first_cp]  * POPULATION_SIZE  # <--- USES CORRECT CHECKPOINT
    laps_completed         = [0]  * POPULATION_SIZE

    episode_rewards        = [0.0]  * POPULATION_SIZE
    steps_in_episode       = [0]    * POPULATION_SIZE
    prev_states            = [None] * POPULATION_SIZE
    prev_actions           = [None] * POPULATION_SIZE
    episode_points_history = [[]  ] * POPULATION_SIZE
    episode_time           = [0.0]  * POPULATION_SIZE
    alive_flags            = [True] * POPULATION_SIZE
    done_reasons           = [None] * POPULATION_SIZE

    episode_count            = 0
    best_episode_max_points  = float('-inf')
    best_episode_end_points  = float('-inf')

    evo_system = EvolutionSystem(population_size=POPULATION_SIZE) # <--- ADD THIS

    current_points         = [float(INITIAL_POINTS)] * POPULATION_SIZE
    checkpoints_passed     = [0]  * POPULATION_SIZE
    current_checkpoint_idx = [0]  * POPULATION_SIZE
    laps_completed         = [0]  * POPULATION_SIZE
    
    episode_fitnesses      = [0.0] * POPULATION_SIZE # <--- ADD THIS


    def reset_episode(episode_count, demo_mode=False):
        # 1. SCOPE DECLARATIONS (Combining your nonlocals with the track globals)
        global outerwall, innerwall, centerline, all_walls, occ_grid, _base_checkpoints
        nonlocal current_points, checkpoints_passed, current_checkpoint_idx
        nonlocal episode_rewards, steps_in_episode, prev_states, prev_actions
        nonlocal episode_points_history, episode_time, alive_flags, done_reasons
        nonlocal episode_fitnesses, current_track_name, agents 

        # =================================================================
        # 2. DYNAMIC TRACK LOADING (The New ML Integration)
        # =================================================================

        global current_track_path
        
        # Change map every 10 episodes, OR if it's the very first episode, OR if in Demo Mode
        
        
        # if episode_count < 300:
        #     difficulty = "easy"
        # elif episode_count < 800:
        #     difficulty = "medium"
        # else:
        #     difficulty = "hard"

        # split = "test" if demo_mode else "train"
        # if demo_mode:
        #     difficulty = "hard" 
        # ── DYNAMIC THRESHOLDS & ROTATION SCALING ──
        if episode_count < 20:
            difficulty = "medium"
            rotation_rate = 5
        # elif episode_count < 300:
        #     difficulty = "medium"
        #     rotation_rate = 8
        else:
            difficulty = "hard"
            rotation_rate = 10

        split = "test" if demo_mode else "train"
        if demo_mode:
            difficulty = "hard" 

        # ── EPSILON SHOCK (EXPLORATION RESET) ──
        # Inject exploration back into the neural nets when they hit a harder difficulty
        if episode_count == 30 or episode_count == 300:
            EPSILON_BY_DIFFICULTY = {
                "medium": 0.30,   # Moderate exploration for medium
                "hard":   0.50,   # High exploration for complex F1 hairpins
            }
            new_eps = EPSILON_BY_DIFFICULTY[difficulty]
            for a in agents:
                a.epsilon = max(a.epsilon, new_eps)
            print(f"\n[CURRICULUM] Switched to {difficulty.upper()} mode! Epsilon reset to {new_eps}\n")        

        if episode_count % rotation_rate == 0 or current_track_path is None or demo_mode:
            # track_path = get_random_track(mode=split, difficulty=difficulty)
            if demo_mode and DEMO_TRACK_PATH:
                track_path = DEMO_TRACK_PATH
            else:
                track_path = get_random_track(mode=split, difficulty=difficulty)
            current_track_path = track_path
            
            # Reload track geometry and rebuild physics
            outerwall, innerwall, centerline = load_and_center_track(track_path)
            current_track_name = os.path.basename(track_path)
            all_walls = np.vstack([outerwall, innerwall])
            track_width = float(np.mean([np.linalg.norm(outerwall[i, :2] - innerwall[i, :2]) for i in range(10)]))
            _base_checkpoints = create_checkpoints_from_centerline(centerline, track_width=track_width, spacing_pixels=120)
            occ_grid = build_occupancy_grid(all_walls, _base_checkpoints)
            
            # Rebuild Checkpoints for the new track


            # car_checkpoints = [copy.deepcopy(_base_checkpoints) for _ in range(POPULATION_SIZE)]


        start_x, start_y, start_angle = None, None, None
        
        while start_x is None:
            # Attempt to find a spawn on the current centerline
            for seg0 in centerline:
                cx = (seg0[0] + seg0[2]) / 2
                cy = (seg0[1] + seg0[3]) / 2
                ang = math.atan2(seg0[3] - seg0[1], seg0[2] - seg0[0])
                
                # Fix 1: Add a 10-pixel buffer to the initial spawn check
                test_pos = np.array([cx + math.cos(ang) * 10, cy + math.sin(ang) * 10])
                
                # Perform a static width-check using the swept capsule engine
                hit_type, _ = swept_capsule_query(test_pos, test_pos, ang, occ_grid)
                
                if hit_type != 'wall':
                    # Fix 2: Ensure cars can survive at least 3 frames moving forward
                    if track_is_valid(cx, cy, ang, occ_grid):
                        start_x, start_y, start_angle = cx, cy, ang
                        break
            
            # If we looped through the entire centerline and found NO safe spawn:
            if start_x is None:
                print(f"[WARN] Track {current_track_name} is a death trap! Force-reloading a new track...")
                
                # Force load a completely new track
                # track_path = get_random_track(mode=split, difficulty=difficulty)
                if demo_mode and DEMO_TRACK_PATH:
                    track_path = DEMO_TRACK_PATH
                else:
                    track_path = get_random_track(mode=split, difficulty=difficulty)
                outerwall, innerwall, centerline = load_and_center_track(track_path)
                current_track_path = track_path
                current_track_name = os.path.basename(track_path)
                all_walls = np.vstack([outerwall, innerwall])
                track_width = float(np.mean([np.linalg.norm(outerwall[i, :2] - innerwall[i, :2]) for i in range(10)]))
                _base_checkpoints = create_checkpoints_from_centerline(centerline, track_width=track_width, spacing_pixels=120)
                occ_grid = build_occupancy_grid(all_walls,_base_checkpoints)
                
                # Rebuild checkpoints for this new forced track
                

                # car_checkpoints = [copy.deepcopy(_base_checkpoints) for _ in range(POPULATION_SIZE)]
                
                # The loop will naturally restart and check this new track for safety.

        # Find the correct starting checkpoint for the validated track
        first_cp = find_first_checkpoint_ahead(_base_checkpoints, start_x, start_y, start_angle)
        # =================================================================
        # 3. POPULATION RESET (Your Exact Code)
        # =================================================================
        for car in cars:
            car.position = np.array([start_x, start_y], dtype=float)
            car.angle    = start_angle
            car.velocity = 0.0
            car.steering = 0.0

        current_points         = [float(INITIAL_POINTS)] * POPULATION_SIZE
        checkpoints_passed     = [0]  * POPULATION_SIZE
        
        # (Optional: Reset to 0 instead of 4 if the new checkpoints list is built from scratch)
        # current_checkpoint_idx =[0]  * POPULATION_SIZE
        current_checkpoint_idx = [first_cp]  * POPULATION_SIZE  # <--- USES CORRECT CHECKPOINT 

        # Note: reset_checkpoints(checkpoints) is safely removed here because 
        # we just generated a brand new, fresh list of checkpoints in Step 2!

        episode_rewards        = [0.0]  * POPULATION_SIZE
        episode_fitnesses      = [0.0]  * POPULATION_SIZE 
        steps_in_episode       = [0]    * POPULATION_SIZE
        prev_states            = [None] * POPULATION_SIZE
        prev_actions           = [None] * POPULATION_SIZE
        episode_points_history = [[]  ] * POPULATION_SIZE
        episode_time           = [0.0]  * POPULATION_SIZE
        alive_flags            = [True] * POPULATION_SIZE
        done_reasons           = [None] * POPULATION_SIZE
        
        if DEMO_MODE:
            alive_flags = [True]  # ensure not killed on first step

    # ------------------------------------------------------------------
    # Pygame
    # ------------------------------------------------------------------
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2D Path Optimiser")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont('Arial', FONT_SIZE_NORMAL)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    global_step = 0
    running = True
    while running:
        global_step += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys_pressed = pygame.key.get_pressed()
        # Change this:
        # dt = clock.get_time() / 1000.0 
        # To this:
        dt = FIXED_DT
        ray_debug    = [None] * POPULATION_SIZE

        # Track first car's ray data for HUD labels
        ray_distances_hud = None
        ray_origin_hud    = None

        for i, car in enumerate(cars):
            if not alive_flags[i]:
                continue

            if TRAINING_MODE or DEMO_MODE:
                origin_pre  = get_car_front(car.position, car.angle)
                rays_pre    = [raycast_dda(origin_pre[0], origin_pre[1],
                                       car.angle + off,
                                       occ_grid)
                               for off in RAY_ANGLES]
                current_state = get_state(car, rays_pre)
                action        = agents[i].act(current_state, training=TRAINING_MODE)
                keys          = agents[i].get_action_keys(action)
            else:
                keys = {
                    'W': bool(keys_pressed[pygame.K_w]),
                    'A': bool(keys_pressed[pygame.K_a]),
                    'S': bool(keys_pressed[pygame.K_s]),
                    'D': bool(keys_pressed[pygame.K_d]),
                }
                action        = None
                current_state = None

            prev_pos   = car.position.copy()
            prev_angle = car.angle
            car.update(keys)

            # # Reward
            # step_reward = 0.0
            # done        = False
            # done_reason = None

            # # Checkpoint logic
            # checkpoint_passed = False
            # ci = current_checkpoint_idx[i]
            # if ci < len(car_checkpoints[i]):
            #     if car_checkpoints[i][ci].check_car_passage(
            #             car.position, car.angle, prev_pos, prev_angle):
            #         current_points[i]  += CHECKPOINT_REWARD
            #         checkpoints_passed[i] += 1
            #         current_checkpoint_idx[i] += 1
            #         checkpoint_passed = True
            #         if current_checkpoint_idx[i] >= len(car_checkpoints[i]):
            #             laps_completed[i]      += 1
            #             current_checkpoint_idx[i] = 0
            #             reset_checkpoints(car_checkpoints[i])

            #             # ── LAP CAP ──────────────────────────────────────────
            #             if laps_completed[i] >= 3:
            #                 if TRAINING_MODE:
            #                     step_reward += 500.0   # big completion bonus
            #                 alive_flags[i]  = False
            #                 done_reasons[i] = "completed"
            #             # ─────────────────────────────────────────────────────

            # current_points[i] -= POINT_DECAY_RATE * dt
            # if TRAINING_MODE:
            #     episode_time[i] += dt

            # # # Reward
            # # step_reward = 0.0
            # # done        = False
            # # done_reason = None

            # if TRAINING_MODE:
            #     # 1. Base positive reward
            #     BASE_REWARD = 1.0
            #     step_reward = BASE_REWARD
                
            #     # 2. Time decay: multiply down instead of subtracting
            #     TIME_PENALTY_FACTOR = max(0.0, 1.0 - POINT_DECAY_RATE * dt)
            #     # TIME_PENALTY_FACTOR = 0.99  # small fixed decay, not zero
            #     step_reward *= TIME_PENALTY_FACTOR
                
            #     # 3. Add velocity bonus
            #     step_reward += max(0.0, car.velocity * 0.1)
                
            #     # 4. Add checkpoint bonus
            #     if checkpoint_passed:
            #         step_reward += CHECKPOINT_REWARD

            # # Collision detection
            #     # nearby = query_spatial_grid(wall_grid, all_walls,
            #     #                             car.position[0], car.position[1],
            #     #                             radius=CAR_QUERY_RADIUS)

            # # Raycasting (post-move)
            # origin_local = get_car_front(car.position, car.angle)
            # rays_local   = [raycast_dda(origin_local[0], origin_local[1],
            #                         car.angle + off,
            #                         occ_grid)
            #                 for off in RAY_ANGLES]
            # ray_debug[i] = (origin_local, car.angle, rays_local)

            # if ray_distances_hud is None:
            #     ray_distances_hud = rays_local
            #     ray_origin_hud    = origin_local

            # # Apply collision penalty (Multiplicative crush)
            # # if car_collides(car.position, car.angle, nearby):
            # if car_collides_occ(car.position, car.angle, occ_grid):
            #     if TRAINING_MODE:
            #         step_reward *= 0.0  # Severe multiplicative crush floors reward at 0
            #     done        = True
            #     done_reason = "collision"

            # if current_points[i] < 0:
            #     done        = True
            #     done_reason = "points_depleted"

            # ==========================================================
            # ── UNIFIED PHYSICS ENGINE (Swept Capsule) ──
            # ==========================================================
            step_reward = 0.0
            done        = False
            done_reason = None
            checkpoint_passed = False

            # ONE query checks for BOTH walls and checkpoints!
            hit_type, hit_idx = swept_capsule_query(prev_pos, car.position, car.angle, occ_grid)

            # 1. Handle Wall Collisions
            if hit_type == 'wall':
                done = True
                done_reason = "collision"
                if TRAINING_MODE:
                    step_reward *= 0.0

            # 2. Handle Checkpoint Crossings
            elif hit_type == 'checkpoint':
                expected_cp = current_checkpoint_idx[i]
                
                # Enforce sequential order: They only get points if they hit the CORRECT next checkpoint
                if hit_idx == expected_cp:   
                    current_points[i] += CHECKPOINT_REWARD
                    checkpoints_passed[i] += 1
                    current_checkpoint_idx[i] += 1
                    checkpoint_passed = True
                    
                    # Did they complete a lap?
                    if current_checkpoint_idx[i] >= len(_base_checkpoints):
                        laps_completed[i] += 1
                        current_checkpoint_idx[i] = 0  # Wrap back to checkpoint 0
                        
                        # Did they finish the race?
                        if laps_completed[i] >= 3:
                            if TRAINING_MODE:
                                step_reward += 500.0
                            done = True
                            done_reason = "completed"

            # ==========================================================
            # ── POINT DECAY & SURVIVAL REWARDS ──
            # ==========================================================
            current_points[i] -= POINT_DECAY_RATE * dt
            
            if TRAINING_MODE:
                episode_time[i] += dt
                # Base velocity reward (only if they didn't hit a wall this frame)
                if not done: 
                    step_reward += max(0.0, car.velocity * 0.2)
                if checkpoint_passed:
                    step_reward += CHECKPOINT_REWARD

            if current_points[i] < 0:
                done        = True
                done_reason = "points_depleted"
                
            # ==========================================================
            # ── POST-MOVE RAYCASTING (For Debug Visuals & Next State) ──
            # ==========================================================
            origin_local = get_car_front(car.position, car.angle)
            rays_local   = [raycast_dda(origin_local[0], origin_local[1],
                                    car.angle + off,
                                    occ_grid)
                            for off in RAY_ANGLES]
            ray_debug[i] = (origin_local, car.angle, rays_local)

            if ray_distances_hud is None:
                ray_distances_hud = rays_local
                ray_origin_hud    = origin_local

            # Store & train
            if TRAINING_MODE and prev_states[i] is not None:
                origin_post = get_car_front(car.position, car.angle)
                rays_post   = [raycast_dda(origin_post[0], origin_post[1],
                                       car.angle + off,
                                       occ_grid)
                               for off in RAY_ANGLES]
                next_state = get_state(car, rays_post)
                agents[i].remember(prev_states[i], prev_actions[i],
                                   step_reward, next_state, done)
                agents[i].replay()
                episode_rewards[i]  += step_reward
                episode_fitnesses[i] += step_reward # <--- ADD THIS
                steps_in_episode[i] += 1

            if TRAINING_MODE:
                prev_states[i]  = current_state
                prev_actions[i] = action
                episode_points_history[i].append(current_points[i])

            if done:
                alive_flags[i]  = False
                done_reasons[i] = done_reason
                if DEMO_MODE:
                    # Respawn immediately
                    cars[i].position = np.array([start_x, start_y], dtype=float)
                    cars[i].angle    = start_angle
                    cars[i].velocity = 0.0
                    cars[i].steering = 0.0
                    alive_flags[i]              = True
                    current_points[i]           = float(INITIAL_POINTS)
                    current_checkpoint_idx[i]   = first_cp
                    checkpoints_passed[i]       = 0
                    # reset_checkpoints(car_checkpoints[i])
        # Episode end
        if TRAINING_MODE and not DEMO_MODE and all(not f for f in alive_flags):
            
            avg_ep_time = np.mean(episode_time)
            if avg_ep_time < FIXED_DT * 5:  # died within 2 frames
                print(f"[SKIP] Degenerate track {current_track_name}. Blacklisting.")
                # Don't evolve, don't count episode, just reload
                for a in agents:
                    a.epsilon = max(a.epsilon, 0.15)
                global current_track_path
                current_track_path = None  # Forces reset_episode to pull a brand new map!
                reset_episode(episode_count, DEMO_MODE)
                continue

            episode_count += 1
            
            track_display_name = os.path.basename(current_track_path) if current_track_path else "Unknown"

            # 1. Print Enhanced Metrics to Console
            print_enhanced_metrics(episode_count, episode_fitnesses, 
                                    checkpoints_passed, laps_completed, 
                                    agents, episode_time, done_reasons, len(_base_checkpoints), current_track_name)
            
            # 2. Identify the absolute best agent
            best_idx = int(np.argmax(episode_fitnesses))
            best_fitness = episode_fitnesses[best_idx]
            avg_fitness = np.mean(episode_fitnesses)
            
            # 3. Save the best agent for Demo Mode
            best_path = os.path.join(MODEL_SAVE_PATH, "best_agent.pth")
            agents[best_idx].save(best_path)
            
            # 4. Save entire population occasionally
            if episode_count % 10 == 0:
                for idx, a in enumerate(agents):
                    save_path = os.path.join(MODEL_SAVE_PATH, f"car_agent_{idx}.pth")
                    a.save(save_path)
            
            # 5. Log metrics to CSV for the Thesis graphs
            file_exists = os.path.isfile(CSV_LOG_PATH)
            with open(CSV_LOG_PATH, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(["Episode", "Best_Fitness", "Avg_Fitness", "Avg_Epsilon", "Best_Checkpoints"])
                
                avg_eps = np.mean([a.epsilon for a in agents])
                writer.writerow([episode_count, best_fitness, avg_fitness, avg_eps, checkpoints_passed[best_idx]])
            
            # 6. Evolve the population
            evo_system.evolve(agents, episode_fitnesses) 

            # Reset the world for the next race in BOTH modes
            reset_episode(episode_count, DEMO_MODE)
            if DEMO_MODE:
                alive_flags = [True]

        # ----------------------------------------------------------------
        # Draw
        # ----------------------------------------------------------------
        if not TRAINING_MODE or (global_step % RENDER_EVERY == 0):
            screen.fill(WHITE)

            wall_thickness = max(1, int(round(2 * S)))
            cl_thickness   = max(1, int(round(2 * S)))

            for seg in outerwall:
                x1, y1, x2, y2 = seg.astype(int)
                pygame.draw.line(screen, BLACK, (x1, y1), (x2, y2), wall_thickness)

            for seg in innerwall:
                x1, y1, x2, y2 = seg.astype(int)
                pygame.draw.line(screen, BLACK, (x1, y1), (x2, y2), wall_thickness)

            for seg in centerline:
                x1, y1, x2, y2 = seg.astype(int)
                pygame.draw.line(screen, RED, (x1, y1), (x2, y2), cl_thickness)

            # Checkpoints – highlight the next one for ANY alive car in green
            # 1. Create a set of all checkpoints currently targeted by living cars
            active_checkpoints = {
                current_checkpoint_idx[i] 
                for i, alive in enumerate(alive_flags) 
                if alive and current_checkpoint_idx[i] < len(_base_checkpoints)
            }

            # 2. Draw the checkpoints (car 0 only — drawing all 50 kills FPS)
            for j, cp in enumerate(_base_checkpoints):
                if j in active_checkpoints:
                    # Draw active checkpoints in GREEN
                    pygame.draw.line(screen, GREEN,
                                    (int(cp.point1[0]), int(cp.point1[1])),
                                    (int(cp.point2[0]), int(cp.point2[1])),
                                    max(2, int(round(5 * S))))
                else:
                    # Draw inactive checkpoints normally
                    cp.draw(screen)

            # Cars
            for i, car in enumerate(cars):
                color = CAR_COLOR_ALIVE if alive_flags[i] else CAR_COLOR_DEAD
                car.draw(screen, color)

            # Debug overlay
            if VISUALIZE_DEBUG:
                for i, data in enumerate(ray_debug):
                    if data is None:
                        continue
                    origin, angle, dists = data
                    for off, dist in zip(RAY_ANGLES, dists):
                        ang = angle + off
                        ex  = origin[0] + math.cos(ang) * dist
                        ey  = origin[1] + math.sin(ang) * dist
                        t   = min(dist / RAY_QUERY_RADIUS, 1.0)
                        col = (int(255 * (1 - t)), int(255 * t), 0)
                        pygame.draw.line(screen, col, origin, (int(ex), int(ey)), 2)
                        pygame.draw.circle(screen, (255, 80, 80),
                                        (int(ex), int(ey)),
                                        max(2, int(round(3 * S))))

                    ax, ay, bx, by = get_car_capsule(cars[i].position, cars[i].angle)
                    draw_capsule(screen, ax, ay, bx, by, CAR_RADIUS, color=(0, 200, 0))

                if ray_distances_hud is not None and ray_origin_hud is not None:
                    labels = ["Front", "R45", "L45", "R90", "L90"]
                    x_label = max(6, int(10 * SW))
                    y_label = int(HEIGHT * 0.55)
                    for k, (lbl, dist) in enumerate(zip(labels, ray_distances_hud)):
                        txt = font.render(f"{lbl}: {dist:.1f}", True, BLACK)
                        screen.blit(txt, (x_label, y_label + k * (FONT_SIZE_NORMAL + 4)))

            # HUD
            draw_hud(screen, font, cars, alive_flags, current_points,
                    checkpoints_passed, laps_completed, episode_count,
                    episode_rewards, agents,len(_base_checkpoints))

            pygame.display.flip()

        clock.tick(0)

    pygame.quit()


if __name__ == "__main__":
    main()