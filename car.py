import numpy as np
import math
import pygame

class Car:
    def __init__(self, position, angle=0.0, turning_circle=1.0):
        """
        Initialize the car at a given position and angle.
        Args:
            position: (x, y) tuple for car's starting position
            angle: initial orientation in radians
        """
        self.position = np.array(position, dtype=float)
        self.angle = angle
        self.velocity = 0.0
        self.steering = 0.0
        self.max_velocity = 5.0
        self.acceleration = 0.2
        self.friction = 0.025
        self.steering_speed = 0.01
        self.drift_factor = 0.02  # Lower = more drift/slippery
        self.turning_circle = turning_circle  # Lower = tighter turns
        
        # Vision system parameters
        self.ray_count = 5
        self.ray_length = 300.0  # Maximum ray length
        self.ray_angles = [0, math.pi/4, -math.pi/4, math.pi/2, -math.pi/2]  # Front, FR, FL, R, L
        self.ray_colors = [(255, 0, 0), (255, 165, 0), (255, 165, 0), (0, 0, 255), (0, 0, 255)]
        self.ray_distances = [0.0] * self.ray_count
        self.ray_endpoints = [(0, 0)] * self.ray_count

    def update(self, keys, track):
        # WASD controls
        if keys.get('W', False):
            self.velocity += self.acceleration
        if keys.get('S', False):
            self.velocity -= self.acceleration
        self.velocity *= (1 - self.friction)
        self.velocity = max(-self.max_velocity, min(self.velocity, self.max_velocity))

        if keys.get('A', False):
            self.steering -= self.steering_speed * self.turning_circle
        elif keys.get('D', False):
            self.steering += self.steering_speed * self.turning_circle
        else:
            self.steering = 0.0

        # Update car angle based on steering and velocity
        self.angle += self.steering * (self.velocity / self.max_velocity)

        # Calculate new position
        direction = np.array([math.cos(self.angle), math.sin(self.angle)])
        new_position = self.position + direction * self.velocity

        # Car dimensions
        car_length, car_width = 40, 20

        # Calculate corners of the car at the new position and angle
        def get_car_corners(center, angle):
            dx = car_length / 2
            dy = car_width / 2
            corners = [
                np.array([dx, dy]),
                np.array([dx, -dy]),
                np.array([-dx, -dy]),
                np.array([-dx, dy])
            ]
            rot = np.array([[math.cos(angle), -math.sin(angle)],
                            [math.sin(angle), math.cos(angle)]])
            return [center + rot @ corner for corner in corners]

        # Point-in-polygon test for collision detection
        def point_in_polygon(pt, poly):
            x, y = pt
            inside = False
            n = len(poly)
            px1, py1 = poly[0]
            for i in range(n+1):
                px2, py2 = poly[i % n]
                if y > min(py1, py2):
                    if y <= max(py1, py2):
                        if x <= max(px1, px2):
                            if py1 != py2:
                                xinters = (y - py1) * (px2 - px1) / (py2 - py1 + 1e-9) + px1
                            if px1 == px2 or x <= xinters:
                                inside = not inside
                px1, py1 = px2, py2
            return inside

        # Create a single polygon from both walls (outer wall + reversed inner wall)
        track_polygon = track.outer_wall + list(reversed(track.inner_wall))
        
        # Check if car would collide
        car_corners = get_car_corners(new_position, self.angle)
        collision = not all(point_in_polygon(tuple(corner), track_polygon) for corner in car_corners)
        
        if not collision:
            self.position = new_position
        else:
            self.velocity = 0.0  # Stop car if any corner collides

        # Cast rays to detect walls
        self.cast_rays(track, point_in_polygon)
        
        # Return whether a collision occurred
        return collision

    def cast_rays(self, track, point_in_polygon):
        """
        Cast rays in different directions from the car using the same collision detection
        as the car itself (point_in_polygon).
        """
        ray_names = ["Front", "Front-Right", "Front-Left", "Right", "Left"]
        track_polygon = track.outer_wall + list(reversed(track.inner_wall))
        
        # Car dimensions
        car_length, car_width = 40, 20
        half_length = car_length / 2
        half_width = car_width / 2
        
        for i, ray_angle in enumerate(self.ray_angles):
            # Calculate ray direction based on car angle
            ray_dir_angle = self.angle + ray_angle
            ray_dir = np.array([math.cos(ray_dir_angle), math.sin(ray_dir_angle)])
            ray_dir = ray_dir / np.linalg.norm(ray_dir)  # Normalize direction vector
            
            # Calculate ray starting position based on car edge
            # The offset depends on which ray we're casting
            if i == 0:  # Front ray
                offset = np.array([math.cos(self.angle), math.sin(self.angle)]) * half_length
            elif i == 1:  # Front-Right ray
                front_offset = np.array([math.cos(self.angle), math.sin(self.angle)]) * half_length
                right_offset = np.array([math.cos(self.angle + math.pi/2), math.sin(self.angle + math.pi/2)]) * half_width
                offset = front_offset + right_offset
            elif i == 2:  # Front-Left ray
                front_offset = np.array([math.cos(self.angle), math.sin(self.angle)]) * half_length
                left_offset = np.array([math.cos(self.angle - math.pi/2), math.sin(self.angle - math.pi/2)]) * half_width
                offset = front_offset + left_offset
            elif i == 3:  # Right ray
                offset = np.array([math.cos(self.angle + math.pi/2), math.sin(self.angle + math.pi/2)]) * half_width
            else:  # Left ray
                offset = np.array([math.cos(self.angle - math.pi/2), math.sin(self.angle - math.pi/2)]) * half_width
            
            # Set ray starting position
            ray_start = self.position + offset
            
            # Sample many points along the ray to find where it exits the track
            found_collision = False
            min_distance = self.ray_length
            
            # Step size for sampling - using a small value for precision
            step_size = 2.0  # pixels
            steps = int(self.ray_length / step_size)
            
            # Start from the ray_start position and move outward along the ray
            for step in range(1, steps + 1):
                distance = step * step_size
                test_point = ray_start + ray_dir * distance
                
                # Check if this point is inside the track polygon
                if not point_in_polygon(tuple(test_point), track_polygon):
                    # We've found where the ray exits the track!
                    found_collision = True
                    min_distance = distance
                    break
            
            # Set the endpoint based on the collision or max length
            intersection_point = ray_start + ray_dir * min_distance
                
            # Store results - Note: store the distance from the car center for consistency
            # but use the proper ray_start for the endpoint calculation
            self.ray_distances[i] = min_distance
            self.ray_endpoints[i] = intersection_point
            
            # Print distance to terminal
            # print(f"{ray_names[i]} Ray: {min_distance:.2f} pixels", end="\t")
        # print()  # Newline after printing all ray distances

    def draw(self, surface, colors):
        # Draw the car
        car_length, car_width = 40, 20
        car_rect = pygame.Rect(0, 0, car_length, car_width)
        car_rect.center = self.position
        car_surf = pygame.Surface((car_length, car_width), pygame.SRCALPHA)
        pygame.draw.rect(car_surf, colors['CAR'], car_surf.get_rect())
        rotated = pygame.transform.rotate(car_surf, -math.degrees(self.angle))
        rect = rotated.get_rect(center=self.position)
        surface.blit(rotated, rect)
        
        # Draw vision rays - NOTE: we need to draw from the car's edge now
        for i in range(self.ray_count):
            # Calculate ray starting position based on car edge (same as in cast_rays)
            car_length, car_width = 40, 20
            half_length = car_length / 2
            half_width = car_width / 2
            
            if i == 0:  # Front ray
                offset = np.array([math.cos(self.angle), math.sin(self.angle)]) * half_length
            elif i == 1:  # Front-Right ray
                front_offset = np.array([math.cos(self.angle), math.sin(self.angle)]) * half_length
                right_offset = np.array([math.cos(self.angle + math.pi/2), math.sin(self.angle + math.pi/2)]) * half_width
                offset = front_offset + right_offset
            elif i == 2:  # Front-Left ray
                front_offset = np.array([math.cos(self.angle), math.sin(self.angle)]) * half_length
                left_offset = np.array([math.cos(self.angle - math.pi/2), math.sin(self.angle - math.pi/2)]) * half_width
                offset = front_offset + left_offset
            elif i == 3:  # Right ray
                offset = np.array([math.cos(self.angle + math.pi/2), math.sin(self.angle + math.pi/2)]) * half_width
            else:  # Left ray
                offset = np.array([math.cos(self.angle - math.pi/2), math.sin(self.angle - math.pi/2)]) * half_width
            
            ray_start = self.position + offset
            
            # Draw the ray line from edge
            pygame.draw.line(
                surface, 
                self.ray_colors[i], 
                ray_start, 
                self.ray_endpoints[i], 
                2
            )
            
            # Draw a small circle at the hit point for better visibility
            pygame.draw.circle(
                surface,
                self.ray_colors[i],
                (int(self.ray_endpoints[i][0]), int(self.ray_endpoints[i][1])),
                4
            )
            
            # Draw distance text at the midpoint of each ray
            midpoint = ((ray_start[0] + self.ray_endpoints[i][0])/2,
                        (ray_start[1] + self.ray_endpoints[i][1])/2)
            
            font = pygame.font.SysFont('Arial', 12)
            distance_text = font.render(f"{self.ray_distances[i]:.1f}", True, (0, 0, 0))
            surface.blit(distance_text, midpoint)
