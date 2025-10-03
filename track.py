import pygame
import sys
import math
import numpy as np
import random

# --- Configuration Constants ---
WIDTH, HEIGHT = 1920, 1080

# Colors
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)

# Extended color dictionary
COLORS = {
    'BLACK': BLACK, 'GRAY': GRAY, 'WHITE': WHITE, 'RED': RED, 
    'GREEN': GREEN, 'BLUE': BLUE, 'YELLOW': YELLOW, 
    'CYAN': CYAN, 'MAGENTA': MAGENTA, 'ORANGE': ORANGE, 'PURPLE': PURPLE
}

# Debug mode flag
DEBUG_MODE = True

# --- Track Segment Classes ---

class TrackSegment:
    """Base class for all track segments."""
    def get_points(self):
        """Returns a list of (x, y) points for the segment."""
        raise NotImplementedError

class StraightSegment(TrackSegment):
    """Generates points for a straight track segment."""
    def __init__(self, length, num_points=10):
        self.length = length
        self.num_points = num_points

    def get_points(self):
        points = []
        for i in range(self.num_points):
            x = (self.length * i) / (self.num_points - 1)
            y = 0
            points.append(np.array([x, y]))
        return points

class HairpinSegment(TrackSegment):
    """Generates points for a hairpin turn segment."""
    def __init__(self, radius, num_points=20):
        self.radius = radius
        self.num_points = num_points

    def get_points(self):
        points = []
        center = np.array([0, self.radius])
        for i in range(self.num_points):
            angle = math.pi * i / (self.num_points - 1)  # 180 degrees
            x = center[0] + self.radius * math.cos(angle)
            y = center[1] - self.radius * math.sin(angle)
            points.append(np.array([x, y]))
        return points

class CurveSegment(TrackSegment):
    """Generates points for a curved track segment with specified angle."""
    def __init__(self, radius, angle_degrees, num_points=20):
        self.radius = radius
        self.angle_radians = math.radians(angle_degrees)
        self.num_points = num_points

    def get_points(self):
        points = []
        for i in range(self.num_points):
            t = i / (self.num_points - 1)
            angle = self.angle_radians * t
            x = self.radius * math.sin(angle)
            y = self.radius * (1 - math.cos(angle))
            points.append(np.array([x, y]))
        return points

# --- Track Generator and Drawing Class ---

class TrackGenerator:
    def __init__(self, segments=None):
        """Initialize with a list of segments or empty."""
        self.segments = segments or []
        self.is_closed = False
        # List of colors to use for segments in debug mode
        self.segment_colors = [RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA, ORANGE, PURPLE]
        
    def add_segment(self, segment_to_add, insert_index=-1):
        """Adds a new segment to the track."""
        if insert_index == -1:
            self.segments.append(segment_to_add)
        else:
            self.segments.insert(insert_index, segment_to_add)
        self.is_closed = False  # Adding a segment breaks the closed loop
    
    def remove_segment(self, index):
        """Removes a segment from the track by index."""
        if 0 <= index < len(self.segments):
            del self.segments[index]
            self.is_closed = False
    
    def close_loop(self):
        """Add a proper segment to close the loop if needed."""
        if not self.segments or len(self.segments) < 2 or self.is_closed:
            return
        
        # Generate points for existing segments without smoothing or closing the loop again
        points = self.generate_raw_track_points()
        
        if len(points) < 2:
            return
            
        # Get the first and last point and their directions
        start_point = points[0]
        end_point = points[-1]
        
        if len(points) >= 3:
            start_dir = points[1] - points[0]
            end_dir = points[-1] - points[-2]
            
            # Calculate the angle between the end direction and the direction to start
            to_start_dir = start_point - end_point
            
            if np.linalg.norm(to_start_dir) < 1.0:
                # Points are already very close
                self.is_closed = True
                return
                
            # Normalize vectors for angle calculation
            end_dir = end_dir / np.linalg.norm(end_dir)
            to_start_dir = to_start_dir / np.linalg.norm(to_start_dir)
            
            # Calculate the angle
            dot = np.dot(end_dir, to_start_dir)
            det = end_dir[0] * to_start_dir[1] - end_dir[1] * to_start_dir[0]
            angle_rad = np.arctan2(det, dot)
            angle_deg = np.degrees(angle_rad)
            
            # Decide what kind of segment to add based on the angle
            distance = np.linalg.norm(start_point - end_point)
            
            if abs(angle_deg) < 30:
                # Small angle - use a straight segment
                closing_segment = StraightSegment(length=distance, num_points=max(5, int(distance / 20)))
            else:
                # Larger angle - use a curve
                # Calculate appropriate radius for the curve
                radius = distance / (2 * np.sin(abs(angle_rad) / 2))
                curve_angle = 180 - abs(angle_deg)
                # Use the sign of the angle to determine curve direction
                if angle_rad < 0:
                    curve_angle = -curve_angle
                closing_segment = CurveSegment(radius=abs(radius), angle_degrees=curve_angle, 
                                             num_points=max(10, int(abs(curve_angle) / 5)))
        else:
            # Not enough points to determine direction - use straight segment
            distance = np.linalg.norm(start_point - end_point)
            closing_segment = StraightSegment(length=distance, num_points=max(5, int(distance / 20)))
            
        self.add_segment(closing_segment)
        self.is_closed = True
    
    def smooth_track(self, points, iterations=2, smoothness=0.25):
        """
        Smooth track using Chaikin's corner cutting algorithm.
        
        Args:
            points: List of points to smooth
            iterations: Number of smoothing passes (more = smoother)
            smoothness: Corner cutting ratio (0.0-0.5, higher = smoother)
        Returns:
            Smoothed list of points
        """
        if len(points) < 3:
            return points
            
        # Constrain smoothness to valid range
        smoothness = max(0.0, min(0.5, smoothness))
        
        # Create a copy of the points to work with
        current_points = points.copy()
        
        # For closed tracks, we need to handle the connection between last and first points
        is_closed = np.linalg.norm(points[0] - points[-1]) < 1.0
        
        # Apply multiple iterations of smoothing
        for _ in range(iterations):
            new_points = []
            
            # Process each pair of adjacent points
            for i in range(len(current_points) - 1):
                p1 = current_points[i]
                p2 = current_points[i + 1]
                
                # Create new points by interpolating between original points
                q1 = p1 * (1 - smoothness) + p2 * smoothness
                q2 = p1 * smoothness + p2 * (1 - smoothness)
                
                # Add first interpolated point
                new_points.append(q1)
                
                # Add second interpolated point (except for the last pair)
                if i < len(current_points) - 2 or not is_closed:
                    new_points.append(q2)
            
            # For closed tracks, handle the connection between last and first points
            if is_closed:
                p1 = current_points[-1]
                p2 = current_points[0]
                
                q1 = p1 * (1 - smoothness) + p2 * smoothness
                q2 = p1 * smoothness + p2 * (1 - smoothness)
                
                new_points.append(q1)
                # We don't add q2 since it would be very close to the first point we already added
            
            # Update current points for next iteration
            current_points = new_points
        
        return current_points
    
    def generate_raw_track_points(self):
        """
        Generates the raw track points without smoothing or closing the loop.
        Uses a simpler and more reliable connection algorithm.
        """
        if not self.segments:
            return []
            
        all_points = []
        segment_indices = []  # Store indices where each segment begins
        
        # Add the first segment points
        first_segment = self.segments[0]
        first_points = first_segment.get_points()
        all_points.extend(first_points)
        segment_indices.append((0, len(first_points)))
        
        # Initialize the last position
        if len(first_points) >= 1:
            last_pos = first_points[-1]
        else:
            last_pos = np.array([0, 0])
        
        # Process each subsequent segment with simpler connection logic
        for i in range(1, len(self.segments)):
            segment = self.segments[i]
            points = segment.get_points()
            
            if not points:
                continue  # Skip empty segments
                
            # Calculate the translation needed to connect this segment
            # We translate the first point of this segment to the last point of the previous
            translation = last_pos - points[0]
            
            # Apply translation to all points in this segment
            transformed = [point + translation for point in points[1:]]  # Skip the first point
            
            # Track where this segment begins
            segment_start = len(all_points)
            
            # Add the transformed points to our track
            all_points.extend(transformed)
            
            # Track where this segment ends
            segment_end = len(all_points)
            segment_indices.append((segment_start, segment_end))
            
            # Update the last position for the next segment
            if transformed:
                last_pos = transformed[-1]
        
        # Store segment indices for debug visualization
        self.segment_indices = segment_indices
        
        return all_points
            
    def generate_full_track_points(self, closed=True, smooth=True):
        """Combines all segments into a single list of points with smooth connections."""
        if not self.segments:
            return []
            
        # Close the loop if requested
        if closed and not self.is_closed:
            self.close_loop()
            
        # Generate the raw track points
        all_points = self.generate_raw_track_points()
        
        # Apply smoothing if requested
        if smooth and len(all_points) > 2:
            smoothed_points = self.smooth_track(all_points, iterations=2, smoothness=0.25)
            # Clear segment indices since smoothing changes the points
            self.segment_indices = []
            return smoothed_points
        else:
            return all_points
    
    def draw_debug(self, surface, center_points):
        """Draw debug visualization of the track centerline with different colors per segment."""
        if not center_points or not hasattr(self, 'segment_indices'):
            return
            
        # Draw lines connecting all center points
        scaled_points = [(int(p[0]), int(p[1])) for p in center_points]
        pygame.draw.lines(surface, BLACK, self.is_closed, scaled_points, 2)
        
        # Draw points for each segment with different colors
        for i, (start_idx, end_idx) in enumerate(self.segment_indices):
            color = self.segment_colors[i % len(self.segment_colors)]
            
            # Draw the segment's points in its assigned color
            for j in range(start_idx, end_idx):
                pygame.draw.circle(surface, color, (int(center_points[j][0]), int(center_points[j][1])), 5)
                
            # Draw first and last point of segment slightly larger for clarity
            if start_idx < len(center_points):
                pygame.draw.circle(surface, BLACK, (int(center_points[start_idx][0]), int(center_points[start_idx][1])), 7)
                pygame.draw.circle(surface, color, (int(center_points[start_idx][0]), int(center_points[start_idx][1])), 5)
            
            if end_idx - 1 < len(center_points):
                pygame.draw.circle(surface, BLACK, (int(center_points[end_idx-1][0]), int(center_points[end_idx-1][1])), 7)
                pygame.draw.circle(surface, color, (int(center_points[end_idx-1][0]), int(center_points[end_idx-1][1])), 5)

class Track:
    def __init__(self, center_points=None, track_width=0):
        self.center_points = center_points or []
        self.track_width = track_width
        self.inner_wall = []
        self.outer_wall = []
        
        # Generate walls if center points are provided
        if center_points and track_width > 0:
            self._generate_walls_from_center()
    
    def _generate_walls_from_center(self):
        """Generate inner and outer walls from center points and track width"""
        if not self.center_points or len(self.center_points) < 2:
            return
        
        half_width = self.track_width / 2
        inner_wall = []
        outer_wall = []
        
        for i in range(len(self.center_points)):
            # Calculate tangent vector
            if i == 0:
                p1 = self.center_points[0]
                p2 = self.center_points[1]
                tangent = (p2[0] - p1[0], p2[1] - p1[1])
            elif i == len(self.center_points) - 1:
                p1 = self.center_points[i-1]
                p2 = self.center_points[i]
                tangent = (p2[0] - p1[0], p2[1] - p1[1])
            else:
                p1 = self.center_points[i-1]
                p2 = self.center_points[i+1]
                tangent = (p2[0] - p1[0], p2[1] - p1[1])
            
            # Normalize tangent
            mag = np.sqrt(tangent[0]**2 + tangent[1]**2)
            if mag > 0:
                tangent = (tangent[0] / mag, tangent[1] / mag)
            else:
                tangent = (1, 0)  # Default if no direction
            
            # Calculate normal vector (perpendicular to tangent)
            normal = (-tangent[1], tangent[0])
            
            # Current point
            p = self.center_points[i]
            
            # Offset points
            inner_wall.append((
                p[0] - normal[0] * half_width,
                p[1] - normal[1] * half_width
            ))
            
            outer_wall.append((
                p[0] + normal[0] * half_width,
                p[1] + normal[1] * half_width
            ))
        
        self.inner_wall = inner_wall
        self.outer_wall = outer_wall
    
    def draw(self, surface, colors):
        """Draw the track on the pygame surface"""
        if self.inner_wall and self.outer_wall:
            # Draw inner wall
            pygame.draw.lines(surface, colors['BLACK'], True, self.inner_wall, 2)
            
            # Draw outer wall
            pygame.draw.lines(surface, colors['BLACK'], True, self.outer_wall, 2)
            
            # Draw center line (if requested)
            if hasattr(self, 'draw_center') and self.draw_center and self.center_points:
                pygame.draw.lines(surface, colors['GRAY'], True, self.center_points, 1)
    
    @staticmethod
    def width_for_difficulty(difficulty):
        """Return an appropriate track width based on difficulty"""
        # Narrower tracks for higher difficulties
        return max(30, 80 - (difficulty - 1) * 5)

def render_debug_info(surface, track_gen, center_points):
    """Display debug information on screen"""
    font = pygame.font.SysFont('Arial', 24)
    y_pos = 10
    
    # Display track info
    segments_text = f"Segments: {len(track_gen.segments)}"
    points_text = f"Center Points: {len(center_points)}"
    mode_text = f"Mode: {'Debug' if DEBUG_MODE else 'Normal'}"
    closed_text = f"Track Closed: {'Yes' if track_gen.is_closed else 'No'}"
    
    # Render and display the text
    for text in [segments_text, points_text, mode_text, closed_text]:
        text_surface = font.render(text, True, BLACK)
        surface.blit(text_surface, (10, y_pos))
        y_pos += 30
    
    # Display controls
    controls_text = "Controls: [D] Toggle Debug Mode"
    text_surface = font.render(controls_text, True, BLACK)
    surface.blit(text_surface, (10, HEIGHT - 40))

# --- Main Program Loop ---

def main():
    global DEBUG_MODE
    
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Track Generation")
    clock = pygame.time.Clock()
    
    # Create a track generator with individual segments
    my_track_gen = TrackGenerator()
    
    # Add various segments to create an interesting track
    my_track_gen.add_segment(StraightSegment(length=300, num_points=15))
    # my_track_gen.add_segment(CurveSegment(radius=150, angle_degrees=90, num_points=20))
    my_track_gen.add_segment(StraightSegment(length=200, num_points=10))
    # my_track_gen.add_segment(HairpinSegment(radius=80, num_points=30))
    my_track_gen.add_segment(StraightSegment(length=250, num_points=12))
    my_track_gen.add_segment(CurveSegment(radius=120, angle_degrees=-60, num_points=15))
    
    # Close the loop to connect back to the start
    my_track_gen.close_loop()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Toggle debug mode with 'D' key
                if event.key == pygame.K_d:
                    DEBUG_MODE = not DEBUG_MODE
                # Regenerate track with 'R' key
                elif event.key == pygame.K_r:
                    my_track_gen.is_closed = False
                    my_track_gen.close_loop()

        # --- Generate and draw the final track ---
        center_points = my_track_gen.generate_full_track_points(smooth=not DEBUG_MODE)

        # Scale and center the track to the screen
        if center_points:
            min_x = min(p[0] for p in center_points)
            max_x = max(p[0] for p in center_points)
            min_y = min(p[1] for p in center_points)
            max_y = max(p[1] for p in center_points)
            w = max(1.0, max_x - min_x)
            h = max(1.0, max_y - min_y)
            margin = 200.0
            scale = min((WIDTH - margin) / w, (HEIGHT - margin) / h)
            cx_src = 0.5 * (min_x + max_x)
            cy_src = 0.5 * (min_y + max_y)
            cx_dst, cy_dst = WIDTH / 2.0, HEIGHT / 2.0
            scaled_points = [((px - cx_src) * scale + cx_dst, (py - cy_src) * scale + cy_dst) for (px, py) in center_points]
            
            # Create track with scaled points
            track = Track(scaled_points, track_width=80)
        else:
            track = None
            scaled_points = []
            
        screen.fill(WHITE)
        
        # Draw track or debug visualization
        if track:
            if DEBUG_MODE:
                # Draw debug visualization
                my_track_gen.draw_debug(screen, scaled_points)
            else:
                # Draw regular track
                track.draw(screen, COLORS)
        
        # Display debug information
        render_debug_info(screen, my_track_gen, scaled_points)
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
