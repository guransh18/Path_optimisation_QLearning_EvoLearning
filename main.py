import pygame
import sys
import pickle
import os
from track import Track
from car import Car
from NeuralNet import DQNAgent, RewardSystem
import numpy as np
import os
# Import checkpoint functionality from Checkpoint.py
from Checkpoint import create_checkpoints, update_checkpoints, draw_checkpoints, reset_checkpoints

WIDTH, HEIGHT = 1920, 1080

# Colors
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
WHITE = (255, 255, 255)
CAR = (255, 0, 0)
YELLOW = (255, 255, 0)
COLORS = {'BLACK': BLACK, 'GRAY': GRAY, 'WHITE': WHITE, 'CAR': CAR, 'YELLOW': YELLOW}

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

def main():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Track & Car Visualization")
    clock = pygame.time.Clock()

    # Let user choose track mode
    track = load_track_from_walls(file_path="Track1.walls", scale_factor=2.0)
    
    if track is None:
        print("Error loading track, exiting...")
        pygame.quit()
        sys.exit()

    start_pos = track.center_points[0]
    car = Car(start_pos, turning_circle=0.4)
    
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
    last_position = car.position.copy()
    
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
                    # Reset manually
                    car.position = np.array(start_pos, dtype=float)
                    car.angle = 0.0
                    car.velocity = 0.0
                    reset_checkpoints(checkpoints)
                    reward_system.reset()
                    time_alive = 0
                    stuck_timer = 0
                
        # Get current state (before update)
        current_state = [
            car.velocity,  # Speed
            car.steering,  # Angular speed/steering
        ] + car.ray_distances  # 5 vision rays
        
        # Choose action based on current state
        action_idx = agent.act(current_state, training=training_mode)
        
        # Convert action to key presses
        keys = agent.get_action_keys(action_idx)
        
        # Store car position and angle before update
        prev_car_position = car.position.copy()
        prev_car_angle = car.angle
                
        # Update car with chosen action
        collision = car.update(keys, track)
        
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
        
        # Check if car is stuck (no significant movement)
        if np.linalg.norm(car.position - last_position) < 0.5 and abs(car.velocity) < 0.1:
            stuck_timer += 1
            if stuck_timer >= stuck_threshold:
                print("Car appears stuck, resetting...")
                episode_finished = True
        else:
            stuck_timer = 0
            last_position = car.position.copy()
        
        # Store experience in agent memory
        if training_mode:
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
            
            # Reset environment
            car.position = np.array(start_pos, dtype=float)
            car.angle = 0.0
            car.velocity = 0.0
            reset_checkpoints(checkpoints)
            reward_system.reset()
            time_alive = 0
            stuck_timer = 0
            
            # Update target network occasionally
            if episode % 10 == 0:
                agent.update_target_model()
        
        # Draw everything
        screen.fill(WHITE)
        track.draw(screen, COLORS)
        
        # Draw checkpoints
        draw_checkpoints(screen, checkpoints)
        
        car.draw(screen, COLORS)
        
        # Display training information
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
