import sys
import numpy as np
import pickle
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, 
                            QRadioButton, QButtonGroup)
from PyQt5.QtGui import QPainter, QPen, QColor, QPainterPath
from PyQt5.QtCore import Qt, QPoint, QPointF

class TrackCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        
        # Initialize drawing properties
        self.drawing = False
        self.last_point = QPoint()
        self.current_path = []
        
        # Track data
        self.inner_wall = []
        self.outer_wall = []
        self.current_wall = 'inner'  # Default to inner wall
        
        # Drawing mode
        self.mode = 'freehand'  # Default to freehand drawing
        self.line_start = None
        
        # History for undo
        self.history = []
        self.save_state()
        
    def save_state(self):
        """Save the current state for undo functionality"""
        state = {
            'inner_wall': [path.copy() if path else [] for path in self.inner_wall],
            'outer_wall': [path.copy() if path else [] for path in self.outer_wall]
        }
        self.history.append(state)
        
    def undo(self):
        """Revert to the previous state"""
        if len(self.history) > 1:
            self.history.pop()  # Remove current state
            previous_state = self.history[-1]
            self.inner_wall = [path.copy() if path else [] for path in previous_state['inner_wall']]
            self.outer_wall = [path.copy() if path else [] for path in previous_state['outer_wall']]
            self.update()
        
    def clear_canvas(self):
        """Clear all drawings"""
        self.inner_wall = []
        self.outer_wall = []
        self.history = []
        self.save_state()
        self.update()
    
    def set_drawing_mode(self, mode):
        """Set the drawing mode (line or freehand)"""
        self.mode = mode
        
    def set_current_wall(self, wall):
        """Set which wall is being drawn (inner or outer)"""
        self.current_wall = wall
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            
            if self.mode == 'line':
                self.line_start = event.pos()
            else:  # freehand
                self.current_path = [self.convert_point(event.pos())]
    
    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.drawing:
            if self.mode == 'freehand':
                self.current_path.append(self.convert_point(event.pos()))
                self.update()
            elif self.mode == 'line':
                # Update the last point for line preview
                self.last_point = event.pos()
                self.update()  # Trigger repaint to show the line preview
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            if self.mode == 'line':
                # Create a line from start to current point
                path = [
                    self.convert_point(self.line_start),
                    self.convert_point(event.pos())
                ]
                if self.current_wall == 'inner':
                    self.inner_wall.append(path)
                else:
                    self.outer_wall.append(path)
                self.line_start = None
            else:  # freehand
                if self.current_wall == 'inner':
                    self.inner_wall.append(self.current_path)
                else:
                    self.outer_wall.append(self.current_path)
                self.current_path = []
            
            self.drawing = False
            self.save_state()
            self.update()
    
    def convert_point(self, qt_point):
        """Convert QPoint to (x,y) tuple"""
        return (qt_point.x(), qt_point.y())
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set background
        painter.fillRect(self.rect(), Qt.white)
        
        # Draw inner wall (blue)
        pen = QPen(Qt.blue, 2)
        painter.setPen(pen)
        self.draw_paths(painter, self.inner_wall)
        
        # Draw outer wall (red)
        pen = QPen(Qt.red, 2)
        painter.setPen(pen)
        self.draw_paths(painter, self.outer_wall)
        
        # Draw current path
        if self.drawing:
            if self.current_wall == 'inner':
                pen = QPen(Qt.blue, 2)
            else:
                pen = QPen(Qt.red, 2)
                
            painter.setPen(pen)
            
            if self.mode == 'line' and self.line_start:
                # Draw line preview from start to current position
                painter.drawLine(self.line_start, self.last_point)
            elif self.mode == 'freehand' and len(self.current_path) > 1:
                path = QPainterPath()
                path.moveTo(self.current_path[0][0], self.current_path[0][1])
                for point in self.current_path[1:]:
                    path.lineTo(point[0], point[1])
                painter.drawPath(path)
    
    def draw_paths(self, painter, paths):
        """Draw a list of paths"""
        for path in paths:
            if len(path) == 2:  # Line
                painter.drawLine(path[0][0], path[0][1], path[1][0], path[1][1])
            elif len(path) > 1:  # Freehand
                qpath = QPainterPath()
                qpath.moveTo(path[0][0], path[0][1])
                for point in path[1:]:
                    qpath.lineTo(point[0], point[1])
                painter.drawPath(qpath)

class TrackEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Track Editor')
        self.setGeometry(100, 100, 1000, 700)
        
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Create drawing canvas
        self.canvas = TrackCanvas()
        
        # Create control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        
        # Wall selection
        wall_group_box = QWidget()
        wall_layout = QHBoxLayout()
        wall_layout.setContentsMargins(0, 0, 0, 0)
        
        self.inner_radio = QRadioButton("Inner Wall")
        self.outer_radio = QRadioButton("Outer Wall")
        self.inner_radio.setChecked(True)
        
        wall_group = QButtonGroup(self)
        wall_group.addButton(self.inner_radio)
        wall_group.addButton(self.outer_radio)
        
        wall_layout.addWidget(self.inner_radio)
        wall_layout.addWidget(self.outer_radio)
        wall_group_box.setLayout(wall_layout)
        
        self.inner_radio.toggled.connect(self.toggle_wall)
        
        # Drawing mode selection
        mode_group_box = QWidget()
        mode_layout = QHBoxLayout()
        mode_layout.setContentsMargins(0, 0, 0, 0)
        
        self.line_radio = QRadioButton("Line")
        self.freehand_radio = QRadioButton("Freehand")
        self.freehand_radio.setChecked(True)
        
        mode_group = QButtonGroup(self)
        mode_group.addButton(self.line_radio)
        mode_group.addButton(self.freehand_radio)
        
        mode_layout.addWidget(self.line_radio)
        mode_layout.addWidget(self.freehand_radio)
        mode_group_box.setLayout(mode_layout)
        
        self.line_radio.toggled.connect(self.toggle_mode)
        
        # Action buttons
        self.undo_btn = QPushButton("Undo")
        self.clear_btn = QPushButton("Clear All")
        self.save_btn = QPushButton("Save Track")
        self.load_btn = QPushButton("Load Track")
        
        # Connect button signals
        self.undo_btn.clicked.connect(self.canvas.undo)
        self.clear_btn.clicked.connect(self.canvas.clear_canvas)
        self.save_btn.clicked.connect(self.save_track)
        self.load_btn.clicked.connect(self.load_track)
        
        # Add widgets to control layout
        control_layout.addWidget(QLabel("Select Wall:"))
        control_layout.addWidget(wall_group_box)
        control_layout.addWidget(QLabel("Drawing Mode:"))
        control_layout.addWidget(mode_group_box)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.undo_btn)
        control_layout.addWidget(self.clear_btn)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.save_btn)
        control_layout.addWidget(self.load_btn)
        control_layout.addStretch(1)
        
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(200)
        
        # Add widgets to main layout
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(control_panel)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    def toggle_wall(self):
        if self.inner_radio.isChecked():
            self.canvas.set_current_wall('inner')
        else:
            self.canvas.set_current_wall('outer')
    
    def toggle_mode(self):
        if self.line_radio.isChecked():
            self.canvas.set_drawing_mode('line')
        else:
            self.canvas.set_drawing_mode('freehand')
    
    def save_track(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Track", "", "Track Files (*.track)")
        if file_path:
            if not file_path.endswith('.track'):
                file_path += '.track'
            
            track_data = {
                'inner_wall': self.canvas.inner_wall,
                'outer_wall': self.canvas.outer_wall
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(track_data, f)
    
    def load_track(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Track", "", "Track Files (*.track)")
        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    track_data = pickle.load(f)
                
                self.canvas.inner_wall = track_data['inner_wall']
                self.canvas.outer_wall = track_data['outer_wall']
                self.canvas.history = []
                self.canvas.save_state()
                self.canvas.update()
            except Exception as e:
                print(f"Error loading track: {e}")

def main():
    app = QApplication(sys.argv)
    editor = TrackEditor()
    editor.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
