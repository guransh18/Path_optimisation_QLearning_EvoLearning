import numpy as np
import pygame
import math
from car import Car

class Checkpoint:
    """
    Checkpoint class to create and manage checkpoints on the track.
    Checkpoints are perpendicular to the track direction.
    """
    def __init__(self, center_point, direction, width, thickness=5, color=(255, 255, 0)):
        """
        Initialize a checkpoint.
        Args:
            center_point: Center point of the checkpoint on the track
            direction: Direction vector along the track (to calculate perpendicular)
            width: Width of the checkpoint line (should be wide enough to cover track width)
            thickness: Thickness of the checkpoint line
            color: Color of the checkpoint line (default: yellow)
        """
        self.center = np.array(center_point)
        self.direction = np.array(direction)
        self.width = width
        self.thickness = thickness
        self.color = color
        self.is_passed = False
        
        # Calculate perpendicular direction
        self.perpendicular = np.array([-direction[1], direction[0]])
        self.perpendicular = self.perpendicular / np.linalg.norm(self.perpendicular)
        
        # Calculate the two endpoints of the checkpoint line
        half_width = self.width / 2
        self.point1 = self.center + self.perpendicular * half_width
        self.point2 = self.center - self.perpendicular * half_width
        
        # Create a small rectangle (polygon) representing the checkpoint
        half_thickness = self.thickness / 2
        forward = self.direction * half_thickness
        backward = -forward
        
        # Define the checkpoint as a polygon (4 corners of a rectangle)
        self.polygon = [
            tuple(self.point1 + forward),
            tuple(self.point1 + backward),
            tuple(self.point2 + backward),
            tuple(self.point2 + forward)
        ]
        
        # For checking if car has passed through
        self.last_car_position = None
    
    def draw(self, surface):
        """Draw the checkpoint line on the surface"""
        if self.is_passed:
            color = (100, 100, 0)  # Darker color for passed checkpoints
        else:
            color = self.color
        
        pygame.draw.line(
            surface,
            color,
            (int(self.point1[0]), int(self.point1[1])),
            (int(self.point2[0]), int(self.point2[1])),
            self.thickness
        )
    
    def check_car_passage(self, car_position, car_angle=None, car_prev_position=None, car_prev_angle=None):
        """
        Check if the car has passed through this checkpoint.
        Uses car's edges to detect crossing.
        """
        if self.is_passed:
            return False
            
        # If we don't have previous position data, save current position and return
        if self.last_car_position is None or car_prev_position is None:
            self.last_car_position = np.array(car_position)
            return False
            
        # Calculate car corners at current and previous positions
        car_length, car_width = 40, 20  # Same dimensions as in Car class
        
        # Get car corners at current position
        current_corners = self.get_car_corners(car_position, car_angle, car_length, car_width)
        
        # Get car corners at previous position
        prev_corners = self.get_car_corners(car_prev_position, car_prev_angle, car_length, car_width)
        
        # Create lines for each car edge (connecting corners)
        car_edges = []
        for i in range(len(current_corners)):
            # Connect previous corner to current corner (movement line)
            car_edges.append((prev_corners[i], current_corners[i]))
            
            # Connect current corners to form car edges
            car_edges.append((current_corners[i], current_corners[(i+1) % len(current_corners)]))
            
        # Checkpoint line
        checkpoint_line = (self.point1, self.point2)
        
        # Check if any car edge intersects with checkpoint line
        for edge in car_edges:
            if self.lines_intersect(edge, checkpoint_line):
                self.is_passed = True
                print("Car hit the checkpoint!")
                return True
                
        # Update the last position for next time
        self.last_car_position = np.array(car_position)
        return False
    
    def get_car_corners(self, position, angle, car_length, car_width):
        """Calculate the four corners of the car based on position and angle"""
        dx = car_length / 2
        dy = car_width / 2
        
        # Create car corners relative to center
        corners = [
            np.array([dx, dy]),
            np.array([dx, -dy]),
            np.array([-dx, -dy]),
            np.array([-dx, dy])
        ]
        
        # Rotate corners based on car angle
        rot = np.array([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)]
        ])
        
        # Apply rotation and translate to car position
        return [position + rot @ corner for corner in corners]
    
    def lines_intersect(self, line1, line2):
        """Check if two line segments intersect"""
        p1, p2 = line1
        p3, p4 = line2
        
        # Convert points to numpy arrays for calculation
        p1, p2, p3, p4 = map(np.array, [p1, p2, p3, p4])
        
        # Calculate direction vectors
        d1 = p2 - p1
        d2 = p4 - p3
        
        # Calculate the determinant
        det = np.cross(d1, d2)
        
        # Lines are parallel if determinant is close to zero
        if abs(det) < 1e-10:
            return False
        
        # Calculate the cross products for parameter values
        t = np.cross(p3 - p1, d2) / det
        u = np.cross(p1 - p3, d1) / -det
        
        # Check if intersection is within both line segments
        return 0 <= t <= 1 and 0 <= u <= 1
    
    def reset(self):
        """Reset the checkpoint status"""
        self.is_passed = False
        self.last_car_position = None


def point_in_polygon(pt, poly):
    """
    Check if a point is inside a polygon using the same algorithm as in car.py.
    Args:
        pt: The point to check (x, y)
        poly: The polygon as a list of points [(x1, y1), (x2, y2), ...]
    Returns:
        True if the point is inside the polygon, False otherwise
    """
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


def calculate_checkpoint_spacing(track, target_count=30):
    """Calculate the appropriate spacing to generate the target number of checkpoints"""
    if not track.center_points:
        return 10  # Default spacing
        
    total_points = len(track.center_points)
    
    # Calculate spacing to get approximately target_count checkpoints
    spacing = max(1, int(total_points / target_count))
    
    return spacing


def calculate_track_length(center_points):
    """Calculate the total length of the track along the center line"""
    length = 0
    for i in range(len(center_points) - 1):
        p1 = np.array(center_points[i])
        p2 = np.array(center_points[i + 1])
        length += np.linalg.norm(p2 - p1)
    
    # Add the connection from last point back to first point for closed tracks
    if len(center_points) > 1:
        p_last = np.array(center_points[-1])
        p_first = np.array(center_points[0])
        if np.linalg.norm(p_last - p_first) < 100:  # Simple heuristic to check if track is closed
            length += np.linalg.norm(p_first - p_last)
    
    return length

def get_point_at_distance(center_points, target_distance):
    """Find the point and direction at a given distance along the track"""
    current_distance = 0
    
    for i in range(len(center_points) - 1):
        p1 = np.array(center_points[i])
        p2 = np.array(center_points[i + 1])
        segment_length = np.linalg.norm(p2 - p1)
        
        if current_distance + segment_length >= target_distance:
            # The target point is on this segment
            # Calculate how far along this segment the point should be
            segment_fraction = (target_distance - current_distance) / segment_length
            point = p1 + segment_fraction * (p2 - p1)
            direction = (p2 - p1) / segment_length  # Normalized direction
            return point, direction
        
        current_distance += segment_length
    
    # If we reach here, return the last point and its direction
    if len(center_points) >= 2:
        p1 = np.array(center_points[-2])
        p2 = np.array(center_points[-1])
        direction = (p2 - p1) / np.linalg.norm(p2 - p1)
        return p2, direction
    
    # Fallback if there are not enough points
    return np.array(center_points[0]), np.array([1.0, 0.0])

def create_checkpoints(track, spacing=None, width_factor=1.5, target_count=30, start_offset=50):
    """
    Create evenly spaced checkpoints along the track based on distance.
    
    Args:
        track: The track object containing center_points
        spacing: Number of points to skip between checkpoints (ignored in distance-based approach)
        width_factor: Factor to multiply track width for checkpoint width
        target_count: Target number of checkpoints
        start_offset: Distance from start position to place the first checkpoint
        
    Returns:
        List of Checkpoint objects
    """
    checkpoints = []
    
    if not track.center_points or len(track.center_points) < 2:
        return checkpoints
    
    # Calculate track length
    track_length = calculate_track_length(track.center_points)
    
    # Calculate checkpoint spacing based on track length and target count
    distance_spacing = track_length / target_count
    
    # Calculate approximate track width if not defined
    if hasattr(track, 'track_width') and track.track_width > 0:
        track_width = track.track_width
    else:
        # Estimate track width from inner and outer walls
        if track.inner_wall and track.outer_wall and len(track.inner_wall) > 0:
            # Sample a few points to estimate width
            sample_count = min(10, len(track.inner_wall))
            sample_indices = [int(i * len(track.inner_wall) / sample_count) for i in range(sample_count)]
            
            widths = []
            for i in sample_indices:
                inner_point = np.array(track.inner_wall[i])
                outer_point = np.array(track.outer_wall[i])
                widths.append(np.linalg.norm(outer_point - inner_point))
            
            track_width = np.mean(widths)
        else:
            # Default width if we can't calculate
            track_width = 80
    
    # Calculate checkpoint width
    checkpoint_width = track_width * width_factor
    
    # Create checkpoints at equal distance intervals with an offset from the start
    for i in range(target_count):
        # Add start_offset to the target distance
        target_distance = i * distance_spacing + start_offset
        
        # Handle the case where target_distance exceeds track length by wrapping around
        target_distance = target_distance % track_length
        
        point, direction = get_point_at_distance(track.center_points, target_distance)
        
        # Create checkpoint at this position
        checkpoint = Checkpoint(point, direction, checkpoint_width)
        checkpoints.append(checkpoint)
    
    # Print diagnostic info
    print(f"Created {len(checkpoints)} checkpoints evenly spaced along track length {track_length:.1f}")
    
    return checkpoints


def update_checkpoints(car, checkpoints, car_prev_position=None, car_prev_angle=None):
    """
    Check if the car has passed through any checkpoints.
    
    Args:
        car: The car object with position attribute
        checkpoints: List of Checkpoint objects
        car_prev_position: Previous car position (for edge detection)
        car_prev_angle: Previous car angle (for edge detection)
        
    Returns:
        Number of checkpoints passed in this update
    """
    passed_count = 0
    for checkpoint in checkpoints:
        if checkpoint.check_car_passage(car.position, car.angle, car_prev_position, car_prev_angle):
            passed_count += 1
    return passed_count


def draw_checkpoints(surface, checkpoints):
    """Draw all checkpoints on the given surface"""
    for checkpoint in checkpoints:
        checkpoint.draw(surface)


def reset_checkpoints(checkpoints):
    """Reset all checkpoints to not passed state"""
    for checkpoint in checkpoints:
        checkpoint.reset()
