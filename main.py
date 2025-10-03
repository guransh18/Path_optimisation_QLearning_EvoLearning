import pygame
import sys
import pickle
import os
from track import Track
from car import Car
import tkinter as tk
from tkinter import filedialog

WIDTH, HEIGHT = 1920, 1080

# Colors
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
WHITE = (255, 255, 255)
CAR = (255, 0, 0)
COLORS = {'BLACK': BLACK, 'GRAY': GRAY, 'WHITE': WHITE, 'CAR': CAR}

def load_track_from_walls(file_path, scale_factor=2.0):
    """Load a track from a saved walls file and scale it by the given factor"""
    if not os.path.exists(file_path):
        print(f"Track file not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'rb') as f:
            walls_data = pickle.load(f)
        
        inner_wall = walls_data.get('inner_wall', [])
        outer_wall = walls_data.get('outer_wall', [])
        
        # Calculate the center of the track for scaling
        all_points = inner_wall + outer_wall
        if not all_points:
            print("Track file contains no points")
            return None
            
        # Calculate center
        center_x = sum(p[0] for p in all_points) / len(all_points)
        center_y = sum(p[1] for p in all_points) / len(all_points)
        
        # Scale points around the center
        inner_wall_scaled = [
            ((p[0] - center_x) * scale_factor + center_x,
             (p[1] - center_y) * scale_factor + center_y)
            for p in inner_wall
        ]
        
        outer_wall_scaled = [
            ((p[0] - center_x) * scale_factor + center_x,
             (p[1] - center_y) * scale_factor + center_y)
            for p in outer_wall
        ]
        
        # Create track from scaled walls
        track = Track(None, 0)
        track.inner_wall = inner_wall_scaled
        track.outer_wall = outer_wall_scaled
        
        # Calculate center points as average of inner and outer walls
        if inner_wall_scaled and outer_wall_scaled and len(inner_wall_scaled) == len(outer_wall_scaled):
            track.center_points = [
                ((inner[0] + outer[0])/2, (inner[1] + outer[1])/2) 
                for inner, outer in zip(inner_wall_scaled, outer_wall_scaled)
            ]
        else:
            # Fallback center points from outer wall if lengths don't match
            track.center_points = [tuple(point) for point in outer_wall_scaled]
            
        print(f"Loaded and scaled track {scale_factor}x with {len(track.inner_wall)} inner wall points and {len(track.outer_wall)} outer wall points")
        
        # Center the track on the screen
        # Calculate current bounds
        min_x = min(min(p[0] for p in inner_wall_scaled), min(p[0] for p in outer_wall_scaled))
        max_x = max(max(p[0] for p in inner_wall_scaled), max(p[0] for p in outer_wall_scaled))
        min_y = min(min(p[1] for p in inner_wall_scaled), min(p[1] for p in outer_wall_scaled))
        max_y = max(max(p[1] for p in inner_wall_scaled), max(p[1] for p in outer_wall_scaled))
        
        # Calculate offset to center
        offset_x = WIDTH/2 - (min_x + max_x)/2
        offset_y = HEIGHT/2 - (min_y + max_y)/2
        
        # Apply offset
        track.inner_wall = [(p[0] + offset_x, p[1] + offset_y) for p in track.inner_wall]
        track.outer_wall = [(p[0] + offset_x, p[1] + offset_y) for p in track.outer_wall]
        track.center_points = [(p[0] + offset_x, p[1] + offset_y) for p in track.center_points]
        
        return track
    except Exception as e:
        print(f"Error loading track: {e}")
        return None

def select_track_mode():
    """Ask user to load a track and set the scale factor"""
    print("Track Loader")
    
    # Scale map by 2x
    scale_factor = 2.0
    
    print(f"Track will be scaled {scale_factor}x")
    
    root = tk.Tk()
    root.withdraw()
 
    file_path= "Track1.walls"
    if file_path:
        return load_track_from_walls(file_path, scale_factor)
    else:
        print("No file selected, exiting...")
        pygame.quit()
        sys.exit()

def main():
	pygame.init()
	pygame.font.init()  # Initialize font system for ray distance display
	screen = pygame.display.set_mode((WIDTH, HEIGHT))
	pygame.display.set_caption("Track & Car Visualization")
	clock = pygame.time.Clock()

	# Let user choose track mode
	track = select_track_mode()
	
	if track is None:
		print("Error loading track, exiting...")
		pygame.quit()
		sys.exit()

	start_pos = track.center_points[0]
	car = Car(start_pos, turning_circle=0.4)

	running = True
	while running:
		keys_pressed = pygame.key.get_pressed()
		keys = {
			'W': keys_pressed[pygame.K_w],
			'A': keys_pressed[pygame.K_a],
			'S': keys_pressed[pygame.K_s],
			'D': keys_pressed[pygame.K_d]
		}
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
		car.update(keys, track)
		screen.fill(WHITE)
		track.draw(screen, COLORS)
		car.draw(screen, COLORS)
		pygame.display.flip()
		clock.tick(60)
	pygame.quit()
	sys.exit()

if __name__ == "__main__":
	main()
