import numpy as np

class Track:
    def __init__(self, center_points, track_width):
        """
        Initialize the track with a list of center points and a width.
        Args:
            center_points: List of (x, y) tuples representing the center line of the track
            track_width: Width of the track
        """
        self.center_points = center_points
        self.track_width = track_width
        self.inner_wall = []
        self.outer_wall = []
        self._calculate_walls()

    def _calculate_walls(self):
        """Calculate the inner and outer walls based on the center line."""
        self.inner_wall = []
        self.outer_wall = []
        num_points = len(self.center_points)
        for i in range(num_points):
            current = np.array(self.center_points[i])
            next_idx = (i + 1) % num_points
            next_point = np.array(self.center_points[next_idx])
            direction = next_point - current
            length = np.linalg.norm(direction)
            if length > 0:
                direction = direction / length
            normal = np.array([-direction[1], direction[0]])
            half_width = self.track_width / 2
            inner_point = current - normal * half_width
            outer_point = current + normal * half_width
            self.inner_wall.append(tuple(inner_point))
            self.outer_wall.append(tuple(outer_point))
        # Ensure the polygon is closed
        self.inner_wall.append(self.inner_wall[0])
        self.outer_wall.append(self.outer_wall[0])

    def draw(self, surface, colors):
        """Draw the track on the given surface. Requires colors dict with 'GRAY', 'BLACK'."""
        import pygame
        # Convert wall points to integers for drawing
        polygon_points = [tuple(map(int, pt)) for pt in self.outer_wall] + [tuple(map(int, pt)) for pt in reversed(self.inner_wall)]
        pygame.draw.polygon(surface, colors['GRAY'], polygon_points)
        pygame.draw.polygon(surface, colors['BLACK'], [tuple(map(int, pt)) for pt in self.outer_wall], 3)
        pygame.draw.polygon(surface, colors['BLACK'], [tuple(map(int, pt)) for pt in self.inner_wall], 3)
