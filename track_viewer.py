import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

# Import the exact logic your neural network uses
from NewTech import load_and_center_track
from Checkpoint import create_checkpoints_from_centerline

# 1. Find all generated tracks in your 80/20 split dataset
track_files = glob.glob("track_dataset/*/*/*.npy")

if not track_files:
    print("[Error] No tracks found. Ensure 'track_dataset' is in this directory.")
    exit()

current_idx = 0
current_spacing = 120 # Default spacing we set

fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.25) # Make extra room for buttons and the new slider

def draw_track(idx):
    ax.clear()
    track_path = track_files[idx]
    
    # 2. Load geometry from the .npy file
    outerwall, innerwall, centerline = load_and_center_track(track_path)
    
    # Estimate track width exactly as NewTech.py does
    n_samp = min(len(outerwall), len(innerwall))
    track_width = float(np.mean([
        np.linalg.norm(outerwall[i, :2] - innerwall[i, :2])
        for i in range(n_samp)
    ]))
    
    # 3. Generate dynamic checkpoints using the NEW relative spacing logic
    checkpoints = create_checkpoints_from_centerline(
        centerline, track_width=track_width, spacing_pixels=current_spacing
    )
    
    # Plot Outer and Inner Walls
    for x1, y1, x2, y2 in outerwall:
        ax.plot([x1, x2], [y1, y2], color='black', lw=2)
    for x1, y1, x2, y2 in innerwall:
        ax.plot([x1, x2], [y1, y2], color='black', lw=2)
        
    # Plot Centerline (Faint)
    for x1, y1, x2, y2 in centerline:
        ax.plot([x1, x2], [y1, y2], color='gray', lw=1, linestyle='--')
        
    # Plot Checkpoints
    for i, cp in enumerate(checkpoints):
        # cp.point1 and cp.point2 are the endpoints of the perpendicular checkpoint line
        ax.plot([cp.point1[0], cp.point2[0]], 
                [cp.point1[1], cp.point2[1]], 
                color='gold', lw=2)
        
        # Highlight the FIRST checkpoint in bright green to show the starting line
        if i == 0:
            ax.plot([cp.point1[0], cp.point2[0]], 
                    [cp.point1[1], cp.point2[1]], 
                    color='lime', lw=4)

    # UI and Formatting
    split_info = track_path.split(os.sep)
    if len(split_info) >= 3:
        title_info = f"Split: {split_info[-3].upper()} | Difficulty: {split_info[-2].upper()}"
    else:
        title_info = ""

    # Dynamically display the total number of checkpoints generated
    ax.set_title(f"Track {idx + 1} / {len(track_files)}\n{os.path.basename(track_path)}\n{title_info}\nTotal Checkpoints: {len(checkpoints)}", fontweight='bold')
    ax.axis('equal')
    ax.invert_yaxis()  # Pygame Y-axis goes down, so we invert Matplotlib to match
    plt.draw()

# --- Interaction Logic ---
def next_track(event):
    global current_idx
    current_idx = (current_idx + 1) % len(track_files)
    draw_track(current_idx)

def prev_track(event):
    global current_idx
    current_idx = (current_idx - 1) % len(track_files)
    draw_track(current_idx)

def update_spacing(val):
    global current_spacing
    current_spacing = spacing_slider.val
    draw_track(current_idx)

# --- UI Elements ---
# Button placement [left, bottom, width, height]
axprev = plt.axes([0.3, 0.05, 0.15, 0.075])
axnext = plt.axes([0.55, 0.05, 0.15, 0.075])
ax_slider = plt.axes([0.3, 0.15, 0.4, 0.03])

bprev = Button(axprev, 'Previous Track')
bnext = Button(axnext, 'Next Track')
# Slider ranges from 50px (very dense) to 400px (very sparse)
spacing_slider = Slider(ax_slider, 'Spacing (px)', 50, 400, valinit=current_spacing, valstep=10)

bprev.on_clicked(prev_track)
bnext.on_clicked(next_track)
spacing_slider.on_changed(update_spacing)

# Draw the first track
draw_track(current_idx)
plt.show()