import pygame
import sys
import pickle
import os
from track import Track
from car import Car
from NeuralNet import DQNAgent, RewardSystem
import numpy as np
import os
# Imprt checkpoint functionaolity from Checkpoint.py
from Checkpoint import create_checkpoints, update_checkpoints, draw_checkpoints, reset_checkpoints

WIDTH, HEIGHT = 1280, 720

# Colors
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
WHITE = (255, 255, 255)
CAR = (255, 0, 0)
YELLOW = (255, 255, 0)
COLORS = {'BLACK': BLACK, 'GRAY': GRAY, 'WHITE': WHITE, 'CAR': CAR, 'YELLOW': YELLOW}



# ===============================================Track Loading & Scaling================================================



def load_track_from_walls(file_path, scale_factor=2.0, track_width=80):
    """Load ONLY the combined/centerline from a saved walls file and scale it by factor 2.0
    
    Optimized to use vectorized numpy operations for better performance.
    Expects a numpy array of shape (N, 4) where each row is (x1, y1, x2, y2) segment.
    
    Args:
        file_path: Path to the .npy file containing segments
        scale_factor: How much to scale the track (default 2.0)
        track_width: Width of the track for generating walls (default 80)
    """
    scale_factor = 2.0  # Force 2.0 scaling as requested

    if not os.path.exists(file_path):
        print(f"Track file not found: {file_path}")
        return None

    try:
        # Load numpy array directly - expecting shape (N, 4) for segments
        segments = np.load(file_path, allow_pickle=True)
        segments = np.asarray(segments, dtype=np.float64).reshape(-1, 4)
        print(f"Loaded {len(segments)} segments from {file_path}")

        if segments.size == 0:
            print("No segment data found in track file")
            return None

        # Vectorized center calculation for all segment endpoints
        all_x = np.concatenate([segments[:, 0], segments[:, 2]])
        all_y = np.concatenate([segments[:, 1], segments[:, 3]])
        center = np.array([all_x.mean(), all_y.mean()])
        
        # Scale segments around center: (point - center) * scale + center
        scaled_segments = segments.copy()
        scaled_segments[:, 0] = (segments[:, 0] - center[0]) * scale_factor + center[0]
        scaled_segments[:, 1] = (segments[:, 1] - center[1]) * scale_factor + center[1]
        scaled_segments[:, 2] = (segments[:, 2] - center[0]) * scale_factor + center[0]
        scaled_segments[:, 3] = (segments[:, 3] - center[1]) * scale_factor + center[1]
        
        # Vectorized bounding box calculation
        min_x = min(scaled_segments[:, 0].min(), scaled_segments[:, 2].min())
        max_x = max(scaled_segments[:, 0].max(), scaled_segments[:, 2].max())
        min_y = min(scaled_segments[:, 1].min(), scaled_segments[:, 3].min())
        max_y = max(scaled_segments[:, 1].max(), scaled_segments[:, 3].max())
        
        # Vectorized centering on screen
        offset_x = WIDTH / 2 - (min_x + max_x) / 2
        offset_y = HEIGHT / 2 - (min_y + max_y) / 2
        
        scaled_segments[:, 0] += offset_x
        scaled_segments[:, 1] += offset_y
        scaled_segments[:, 2] += offset_x
        scaled_segments[:, 3] += offset_y
        
        # Calculate midpoints from SCALED segments for center_points
        midpoints = np.column_stack([
            (scaled_segments[:, 0] + scaled_segments[:, 2]) * 0.5,
            (scaled_segments[:, 1] + scaled_segments[:, 3]) * 0.5
        ])
        
        # Generate inner and outer walls by offsetting perpendicular to centerline
        # This is needed for collision detection in car.py
        n_points = len(midpoints)
        half_width = track_width / 2
        
        inner_wall = []
        outer_wall = []
        
        for i in range(n_points):
            # Calculate tangent direction (vector along the track)
            if i == 0:
                tangent = midpoints[1] - midpoints[0]
            elif i == n_points - 1:
                tangent = midpoints[i] - midpoints[i - 1]
            else:
                tangent = midpoints[i + 1] - midpoints[i - 1]
            
            # Normalize tangent
            tangent_len = np.linalg.norm(tangent)
            if tangent_len > 0:
                tangent = tangent / tangent_len
            else:
                tangent = np.array([1.0, 0.0])
            
            # Normal vector (perpendicular to tangent)
            normal = np.array([-tangent[1], tangent[0]])
            
            # Offset points to create walls
            center_pt = midpoints[i]
            inner_wall.append(tuple(center_pt - normal * half_width))
            outer_wall.append(tuple(center_pt + normal * half_width))
        
        # Create track object
        track = Track(None, 0)
        
        # Convert to list of tuples for compatibility with existing Track class
        track.center_points = [tuple(p) for p in midpoints]
        
        # Set proper walls for collision detection
        track.inner_wall = inner_wall
        track.outer_wall = outer_wall
        
        # Store the scaled segments for drawing
        track._segments_np = scaled_segments
        track._center_points_np = midpoints
        
        print(f"Loaded combined track (centerline) scaled {scale_factor}x with {len(track.center_points)} points")
        print(f"Generated walls with track_width={track_width} for collision detection")
        
        return track
        
    except Exception as e:
        print(f"Error loading track: {e}")
        import traceback
        traceback.print_exc()
        return None


# =================================================DRAWING SPEEDOMETER================================================


def draw_speedometer(surface, car, position=(50, HEIGHT-100)):
    """
    Draw a speedometer with large digits in the bottom left corner.
    
    Args:
        surface: Pygame surface to draw on
        car: Car object with velocity attribute
        position: (x, y) position for the speedometer
    """
    # Calculate absolute speed (both forward and backward)
    speed = abs(car.velocity)
    
    # Convert to km/h or mph for display (arbitrary scale factor)
    speed_display = speed * 20  # Scale factor to make the number more realistic
    
    # Create large font for the speed digits
    speed_font = pygame.font.SysFont('Arial', 72, bold=True)
    unit_font = pygame.font.SysFont('Arial', 36)
    
    # Render speed text with large digits
    speed_text = speed_font.render(f"{speed_display:.1f}", True, BLACK)
    
    # Render "km/h" text with smaller font
    unit_text = unit_font.render("km/h", True, BLACK)
    
    # Calculate positions
    speed_pos = position
    unit_pos = (position[0] + speed_text.get_width() + 10, position[1] + speed_text.get_height() - unit_text.get_height())
    
    # Draw background for better visibility
    bg_rect = pygame.Rect(
        position[0] - 10,
        position[1] - 10,
        speed_text.get_width() + unit_text.get_width() + 30,
        speed_text.get_height() + 20
    )
    pygame.draw.rect(surface, (240, 240, 240), bg_rect, border_radius=10)
    pygame.draw.rect(surface, BLACK, bg_rect, 2, border_radius=10)
    
    # Draw the text
    surface.blit(speed_text, speed_pos)
    surface.blit(unit_text, unit_pos)
    
    # Add a speedometer label
    label_font = pygame.font.SysFont('Arial', 24)
    label_text = label_font.render("SPEED", True, BLACK)
    label_pos = (position[0], position[1] - 30)
    surface.blit(label_text, label_pos)



# =================================================MAIN================================================



def main():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Track & Car Visualization")
    clock = pygame.time.Clock()

    # Let user choose track mode
    track = load_track_from_walls(file_path="Track1_combined_segments.npy", scale_factor=2.0)
    
    if track is None:
        print("Error loading track, exiting...")
        pygame.quit()
        sys.exit()

    # Use the SCALED segments from track object (already transformed)
    combined_segments = None
    if hasattr(track, '_segments_np') and track._segments_np is not None:
        combined_segments = track._segments_np.astype(np.float32)
        print(f"Using pre-loaded SCALED segments from track (shape={combined_segments.shape})")

    # Start position is now from the scaled/centered midpoints
    # Use a point further into the track (not the first point) to avoid edge cases
    start_idx = min(5, len(track.center_points) - 1)  # Start a few points in
    start_pos = track.center_points[start_idx]
    
    # Calculate initial angle based on track direction at start position
    if start_idx + 1 < len(track.center_points):
        next_pos = track.center_points[start_idx + 1]
        dx = next_pos[0] - start_pos[0]
        dy = next_pos[1] - start_pos[1]
        start_angle = np.degrees(np.arctan2(dy, dx))
    else:
        start_angle = 0.0
    
    print(f"Car start position: {start_pos}, angle: {start_angle:.1f}°")
    car = Car(start_pos, turning_circle=0.4)
    car.angle = start_angle  # Set initial angle to face along the track
    
    # Create checkpoints on the track with an offset from the start position
    checkpoints = create_checkpoints(track, spacing=None, width_factor=1.5, target_count=30)
    print(f"Created {len(checkpoints)} checkpoints on the track")
    
    # Initialize font for checkpoint display
    font = pygame.font.SysFont('Arial', 24)
    


    # ================================================Neural Network Agent Setup================================================




    # Initialize neural network agent
    # 7 inputs (speed, angular speed, 5 ray distances), 4 outputs (W, A, S, D)
    agent = DQNAgent(state_size=7, action_size=4, hidden_size=64)
    
    # Load model if available
    model_path = "normal_car.pth"
    if os.path.exists(model_path):
        agent.load(model_path)
    
    # Initialize reward system
    reward_system = RewardSystem(initial_points=100, checkpoint_reward=100, decay_rate=0.2)
    
    # Initialize training variables
    training_mode = True  # Set to False to only use the trained model without updating
    # Manual control toggle: when True, user drives with WASD and training is paused.
    manual_control = False
    prev_training_mode = training_mode  # preserve previous training state when toggling manual control
    episode = 0
    max_episodes = 1000
    time_alive = 0
    update_frequency = 4  # Update network every N frames
    frame_count = 0
    
    # Track previous car position and angle for edge-based checkpoint detection
    prev_car_position = car.position.copy()
    prev_car_angle = car.angle
    
    # For saving best models
    best_checkpoint_count = 0
    
    # For automatic reset when car gets stuck
    stuck_timer = 0
    stuck_threshold = 300  # Reset after 5 seconds (60 fps * 5)
    last_position = np.array(car.position, dtype=np.float64)  # Use numpy for efficient distance calc
    
    # Pre-compute start position as numpy array for fast resets
    start_pos_np = np.array(start_pos, dtype=np.float64)
    
    running = True
    while running and episode < max_episodes:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    # Toggle training mode
                    training_mode = not training_mode
                    print(f"Training mode: {training_mode}")
                elif event.key == pygame.K_s:
                    # Save model manually
                    agent.save(model_path)
                elif event.key == pygame.K_r:
                    # Reset manually using pre-computed numpy start position
                    car.position = start_pos_np.copy()
                    car.angle = start_angle
                    car.velocity = 0.0
                    reset_checkpoints(checkpoints)
                    reward_system.reset()
                    time_alive = 0
                    stuck_timer = 0
                    last_position = start_pos_np.copy()
                elif event.key == pygame.K_m:
                    # Toggle manual control (WASD). Pause/resume training state when entering/exiting manual.
                    manual_control = not manual_control
                    if manual_control:
                        prev_training_mode = training_mode
                        training_mode = False
                    else:
                        training_mode = prev_training_mode
                    print(f"Manual control: {manual_control} | Training mode: {training_mode}")
        
        # Always poll keyboard state once per frame (used for manual control and fallbacks)
        kp = pygame.key.get_pressed()
        
        # Select keys either from the agent or from manual WASD input
        if manual_control:
            # Primary format: pass the pygame key-press sequence (most car.update implementations accept this)
            keys = kp
            # We do not query the agent or store the NN state while in manual control.
            current_state = None
            action_idx = None
        else:
            # Get current state (before update)
            current_state = [
                car.velocity,  # Speed
                car.steering,  # Angular speed/steering
            ] + car.ray_distances  # 5 vision rays
            
            # Choose action based on current state
            action_idx = agent.act(current_state, training=training_mode)
            
            # Convert action to key presses from agent
            keys = agent.get_action_keys(action_idx)
        
        # Store car position and angle before update
        prev_car_position = car.position.copy()
        prev_car_angle = car.angle
                
        # Update car with chosen action
        # Some car.update implementations accept different key formats:
        #  - the pygame.key.get_pressed() sequence
        #  - a dict keyed by pygame.K_* constants
        #  - a dict keyed by 'w','a','s','d' strings
        # Try the primary format first and fall back if an exception occurs.
        try:
            collision = car.update(keys, track)
        except Exception:
            # fallback 1: dict keyed by pygame key constants
            try:
                keys_dict = {
                    pygame.K_w: kp[pygame.K_w],
                    pygame.K_s: kp[pygame.K_s],
                    pygame.K_a: kp[pygame.K_a],
                    pygame.K_d: kp[pygame.K_d],
                }
                collision = car.update(keys_dict, track)
            except Exception:
                # fallback 2: simple string-key dict (some implementations check 'w','a','s','d')
                keys_str = {
                    'w': bool(kp[pygame.K_w]),
                    'a': bool(kp[pygame.K_a]),
                    's': bool(kp[pygame.K_s]),
                    'd': bool(kp[pygame.K_d]),
                }
                collision = car.update(keys_str, track)
        
        # Increment time alive
        time_alive += 1
        
        # Check if car passed through any checkpoints
        update_checkpoints(car, checkpoints, prev_car_position, prev_car_angle)
        
        # Get new state after update
        new_state = [
            car.velocity,
            car.steering,
        ] + car.ray_distances
        
        # Calculate reward
        reward = reward_system.calculate_reward(car, checkpoints, collision, time_alive)
        
        # Check if episode is done (collision or all checkpoints passed)
        episode_finished = collision or reward_system.is_out_of_points() or (sum(cp.is_passed for cp in checkpoints) == len(checkpoints))
        
        # Check if car is stuck (no significant movement) - using numpy for faster distance calc
        displacement = np.linalg.norm(np.asarray(car.position) - last_position)
        if displacement < 0.5 and abs(car.velocity) < 0.1:
            stuck_timer += 1
            if stuck_timer >= stuck_threshold:
                print("Car appears stuck, resetting...")
                episode_finished = True
        else:
            stuck_timer = 0
            last_position = np.array(car.position, dtype=np.float64)
        
        # Store experience in agent memory (only when agent is controlling and training is enabled)
        if (not manual_control) and training_mode:
            agent.remember(current_state, action_idx, reward, new_state, episode_finished)
            
            # Train the model every few frames
            frame_count += 1
            if frame_count % update_frequency == 0:
                agent.replay()
        
        # Reset if episode is done
        if episode_finished:
            episode += 1
            checkpoint_count = sum(cp.is_passed for cp in checkpoints)
            
            # Record episode stats
            agent.episode_rewards.append(reward_system.total_reward)
            agent.checkpoint_rewards.append(checkpoint_count)
            
            print(f"Episode: {episode}, Checkpoints: {checkpoint_count}/{len(checkpoints)}, " +
                  f"Total Reward: {reward_system.total_reward:.1f}, Epsilon: {agent.epsilon:.4f}")
            
            # Save model if this is the best performance
            if checkpoint_count > best_checkpoint_count:
                best_checkpoint_count = checkpoint_count
                agent.save(f"best_car.pth")
                print(f"New best model saved! Checkpoints: {checkpoint_count}")
            
            # Save regularly regardless of performance
            if episode % 10 == 0:
                agent.save(model_path)
            
            # Reset environment using pre-computed numpy array
            car.position = start_pos_np.copy()
            car.angle = start_angle
            car.velocity = 0.0
            reset_checkpoints(checkpoints)
            reward_system.reset()
            time_alive = 0
            stuck_timer = 0
            last_position = start_pos_np.copy()
            
            # Update target network occasionally
            if episode % 10 == 0:
                agent.update_target_model()
        
        # Draw everything
        screen.fill(WHITE)
        
        # Don't call track.draw() - we'll draw the scaled segments directly instead
        # track.draw(screen, COLORS)  # This draws inner/outer walls which we don't need visually

        # Draw the track using scaled segments (centerline visualization)
        if combined_segments is not None and combined_segments.size:
            n = combined_segments.shape[0]
            
            # Pre-compute integer coordinates using vectorized operations
            seg_int = combined_segments.astype(np.int32)
            
            # Draw all segments in a single color (dark gray for track)
            for i in range(n):
                pygame.draw.line(screen, GRAY, 
                               (seg_int[i, 0], seg_int[i, 1]), 
                               (seg_int[i, 2], seg_int[i, 3]), 3)
            
            # Optionally draw start marker (green circle at first segment)
            pygame.draw.circle(screen, (0, 200, 0), (seg_int[0, 0], seg_int[0, 1]), 8)

        # Draw checkpoints
        draw_checkpoints(screen, checkpoints)

        car.draw(screen, COLORS)
        
        # Display mode information (Manual overrides agent modes)
        if manual_control:
            mode_text = "Manual"
        else:
            mode_text = "Training" if training_mode else "Inference"
        info_text = f"Mode: {mode_text} | Episode: {episode} | Reward: {max(0, int(reward_system.remaining_points))}"
        text_surface = font.render(info_text, True, BLACK)
        screen.blit(text_surface, (10, 10))
        
        # Display checkpoint information
        checkpoint_text = f"Checkpoints: {sum(cp.is_passed for cp in checkpoints)}/{len(checkpoints)}"
        text_surface = font.render(checkpoint_text, True, BLACK)
        screen.blit(text_surface, (10, 40))
        
        # Draw speedometer
        draw_speedometer(screen, car)
        
        pygame.display.flip()
        clock.tick(60)
        
    # Save final model
    if training_mode:
        agent.save(model_path)
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
	main()
