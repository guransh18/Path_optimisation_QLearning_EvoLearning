import numpy as np
import math

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

        # Collision detection with track walls (polygon)
        def point_in_polygon(pt, poly):
            # Ray casting algorithm
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

        track_polygon = track.outer_wall + list(reversed(track.inner_wall))
        car_corners = get_car_corners(new_position, self.angle)
        if all(point_in_polygon(tuple(corner), track_polygon) for corner in car_corners):
            self.position = new_position
        else:
            self.velocity = 0.0  # Stop car if any corner collides

    def draw(self, surface, colors):
        import pygame
        car_length, car_width = 40, 20
        car_rect = pygame.Rect(0, 0, car_length, car_width)
        car_rect.center = self.position
        car_surf = pygame.Surface((car_length, car_width), pygame.SRCALPHA)
        pygame.draw.rect(car_surf, colors['CAR'], car_surf.get_rect())
        rotated = pygame.transform.rotate(car_surf, -math.degrees(self.angle))
        rect = rotated.get_rect(center=self.position)
        surface.blit(rotated, rect)
        rotated = pygame.transform.rotate(car_surf, -math.degrees(self.angle))
        rect = rotated.get_rect(center=self.position)
        surface.blit(rotated, rect)
