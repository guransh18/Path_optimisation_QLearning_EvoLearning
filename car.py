import numpy as np
import math
import pygame

# ---------------------------------------------------------------------------
# Reference resolution – the baseline the original constants were tuned for.
# ---------------------------------------------------------------------------
_BASE_W = 1280
_BASE_H = 720


def _scale(width: int, height: int) -> float:
    """Uniform scale factor relative to the 1280×720 baseline (diagonal)."""
    base_diag = math.hypot(_BASE_W, _BASE_H)
    curr_diag = math.hypot(width, height)
    return curr_diag / base_diag


class Car:
    def __init__(self, position, angle=0.0, turning_circle=3,
                 screen_width: int = _BASE_W, screen_height: int = _BASE_H):
        """
        Initialize the car.

        Args:
            position:      (x, y) starting position
            angle:         initial orientation in radians
            turning_circle: steering sensitivity multiplier
            screen_width / screen_height: current window size for scaling
        """
        self.position = np.array(position, dtype=float)
        self.angle = angle
        self.velocity = 0.0
        self.steering = 0.0

        s = _scale(screen_width, screen_height)
        self._scale = s

        # Physics – velocity/acceleration scale with window size; ratios preserved
        self.max_velocity   = 8.0 * s
        self.acceleration   = 0.30  * s
        self.friction       = 0.025      # dimensionless
        self.steering_speed = 0.095      # radians/frame – angular, no pixel scaling
        self.turning_circle = turning_circle

        # Visual / collision dimensions
        self.car_length = max(4, int(round(40 * s)))
        self.car_width  = max(2, int(round(20 * s)))

        # Ray distances placeholder expected by NeuralNet / main loops
        self.ray_distances = [300.0 * s] * 5

    # ------------------------------------------------------------------

    def update(self, keys):
        """Update position from WASD dict."""
        if keys.get('W', False):
            self.velocity += self.acceleration
        if keys.get('S', False):
            self.velocity -= self.acceleration
        self.velocity *= (1 - self.friction)
        self.velocity = max(-self.max_velocity, min(self.velocity, self.max_velocity))

        if keys.get('A', False):
            self.steering = -self.steering_speed * self.turning_circle
        elif keys.get('D', False):
            self.steering =  self.steering_speed * self.turning_circle
        else:
            self.steering = 0.0

        self.angle += self.steering * (self.velocity / self.max_velocity)
        direction = np.array([math.cos(self.angle), math.sin(self.angle)])
        self.position = self.position + direction * self.velocity

    # ------------------------------------------------------------------

    def draw(self, surface, color=(0, 0, 255)):
        """Draw the car rectangle, rotated to face self.angle."""
        car_surf = pygame.Surface((self.car_length, self.car_width), pygame.SRCALPHA)
        pygame.draw.rect(car_surf, color, car_surf.get_rect())
        rotated = pygame.transform.rotate(car_surf, -math.degrees(self.angle))
        rect = rotated.get_rect(center=self.position)
        surface.blit(rotated, rect)