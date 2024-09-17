import sys
import nibabel as nib
import numpy as np
import SimpleITK as sitk  # Ensure SimpleITK is installed for this
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QPushButton, QSlider, QWidget, QLabel, QHBoxLayout, \
    QCheckBox, QScrollArea, QGridLayout, QFileDialog, QSizePolicy, QDesktopWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QPixmap
import os
from app_utils import handle_file_upload, extract_slice, clean_nifti_dir


class NiftiViewer(QMainWindow):
    def __init__(self, initial_file=None):
        super().__init__()

        # Dictionary to store all uploaded files (filename -> NIfTI object)
        self.uploaded_files = {}

        # If there's an initial file, load it
        if initial_file:
            self.load_nifti_file(initial_file)

        self.slice_idx = 0
        self.time_idx = 0
        self.playing = False
        self.dark_mode_enabled = True
        self.playback_speed = 100  # Default playback speed, will be updated for CINE

        self.init_ui()

        # Timer for frame updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        # Get monitor resolution and resize accordingly
        self.screen_resolution = QApplication.desktop().screenGeometry()
        self.screen_width = self.screen_resolution.width()
        self.screen_height = self.screen_resolution.height()

        self.resize(int(self.screen_width * 0.8), int(self.screen_height * 0.8))  # Resize window to 80% of screen size

    def closeEvent(self, a0):
        try:
            clean_nifti_dir()
        except:
            pass
        a0.accept()

    def init_ui(self):
        # Main widget
        widget = QWidget()
        self.setCentralWidget(widget)

        # Main layout
        self.main_layout = QHBoxLayout()
        widget.setLayout(self.main_layout)

        # Left sidebar for thumbnails (scrollable)
        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.scroll_layout = QGridLayout()
        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedWidth(200)  # Set a fixed width for the thumbnail area
        self.main_layout.addWidget(self.scroll_area)

        # Right side for image display and controls
        self.right_layout = QVBoxLayout()
        self.main_layout.addLayout(self.right_layout)

        # Canvas for displaying the image (dynamically resizeable)
        self.figure, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_anchor('C')  # Center the image initially
        self.canvas = FigureCanvas(self.figure)

        # Set the canvas to expand dynamically
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        self.right_layout.addWidget(self.canvas)

        # Add buttons and sliders
        self.add_controls()

        # Button to upload additional files
        self.upload_button = QPushButton("Upload New File")
        self.upload_button.clicked.connect(self.upload_file)
        self.right_layout.addWidget(self.upload_button)

        # Apply default dark mode
        self.apply_dark_mode()

    def resizeEvent(self, event):
        """Dynamically resize the canvas and keep the image centered."""
        canvas_width, canvas_height = self.canvas.width(), self.canvas.height()
        # Resize the figure based on canvas size
        self.figure.set_size_inches(canvas_width / self.figure.dpi, canvas_height / self.figure.dpi, forward=True)
        self.ax.set_anchor('C')  # Keep the image centered
        if hasattr(self, 'img_data'):  # Check if img_data is initialized
            self.update_image()  # Redraw the image to fit the new size

    def add_controls(self):
        # Play/Stop buttons for CINE mode
        button_layout = QHBoxLayout()
        self.play_button = QPushButton('Play')
        self.play_button.clicked.connect(self.toggle_play)
        button_layout.addWidget(self.play_button)

        self.stop_button = QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop_playback)
        button_layout.addWidget(self.stop_button)

        self.right_layout.addLayout(button_layout)

        # Label and slider for playback speed (CINE only)
        self.playback_speed_label = QLabel(f"Playback Speed: {self.playback_speed} ms")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(10)  # Fastest
        self.speed_slider.setMaximum(500)  # Slowest
        self.speed_slider.setValue(self.playback_speed)
        self.speed_slider.setTickInterval(50)
        self.speed_slider.valueChanged.connect(self.adjust_speed)
        self.right_layout.addWidget(self.playback_speed_label)
        self.right_layout.addWidget(self.speed_slider)

        # Label and slider for slice depth (3D mode)
        self.slice_depth_label = QLabel("Slice Depth")
        self.slice_slider = QSlider(Qt.Horizontal)
        self.right_layout.addWidget(self.slice_depth_label)
        self.right_layout.addWidget(self.slice_slider)

        # Add slice/frame indicator label
        self.frame_slice_label = QLabel("Slice 0")  # Initially set to Slice 0
        self.right_layout.addWidget(self.frame_slice_label)

        # Dark Mode Toggle (enabled by default)
        self.dark_mode_checkbox = QCheckBox("Dark Mode")
        self.dark_mode_checkbox.setChecked(True)
        self.dark_mode_checkbox.stateChanged.connect(self.toggle_dark_mode)
        self.right_layout.addWidget(self.dark_mode_checkbox)

        # Initialize controls and labels to be hidden (they will be shown/hidden based on file type)
        self.play_button.setVisible(False)
        self.stop_button.setVisible(False)
        self.speed_slider.setVisible(False)
        self.playback_speed_label.setVisible(False)
        self.slice_slider.setVisible(False)
        self.slice_depth_label.setVisible(False)
        self.frame_slice_label.setVisible(False)

    def load_nifti_file(self, file_path):
        """Load the NIfTI file and update the main image display."""
        try:
            nii_file = nib.load(file_path)
            self.uploaded_files[os.path.basename(file_path)] = nii_file
            self.add_thumbnail(file_path)

            if len(self.uploaded_files) == 1:
                # Load the first file by default
                self.current_file = file_path
                self.img_data = nii_file.get_fdata()
                self.slice_idx = self.img_data.shape[2] // 2  # Default middle slice
                self.update_image()

            # Use SimpleITK to read temporal resolution from the NIfTI header
            itk_img = sitk.ReadImage(file_path)
            img_spacing = itk_img.GetSpacing()

            # If it's a CINE scan (4D image), set the playback speed to the temporal resolution
            if len(self.img_data.shape) == 4:
                temporal_resolution = img_spacing[3] * 1000  # Convert seconds to milliseconds
                self.playback_speed = int(temporal_resolution)  # Set playback speed
                self.playback_speed_label.setText(f"Playback Speed: {self.playback_speed} ms")
                self.speed_slider.setValue(self.playback_speed)

            # Update window title with file name
            self.setWindowTitle(f'NIfTI Viewer - {os.path.basename(file_path)}')

            # Switch between 3D and CINE modes based on file type
            self.switch_mode(nii_file)

        except Exception as e:
            print(f"Error loading file: {file_path}\n{e}")

    def add_thumbnail(self, file_path):
        """Create a thumbnail for the file and add it to the sidebar."""
        try:
            filename = os.path.basename(file_path)
            nii_file = self.uploaded_files[filename]
            middle_slice = extract_slice(nii_file)

            # Create thumbnail image
            fig, ax = plt.subplots(figsize=(1.7, 1.7))
            ax.imshow(middle_slice, cmap='gray')
            ax.axis('off')
            fig.patch.set_facecolor("black")
            plt.tight_layout()

            # Convert the plot to a canvas and use it as a thumbnail
            canvas = FigureCanvas(plt.gcf())
            canvas.draw()

            # Convert canvas to QPixmap and then to QIcon
            thumbnail_pixmap = QPixmap(canvas.grab())
            thumbnail_icon = QIcon(thumbnail_pixmap)

            # Create a button with the thumbnail
            thumbnail_button = QPushButton()
            thumbnail_button.setIcon(thumbnail_icon)  # Set as QIcon, not QPixmap
            thumbnail_button.setIconSize(thumbnail_pixmap.size())  # Adjust the size
            thumbnail_button.clicked.connect(lambda: self.select_file_by_thumbnail(file_path))  # Ensure mode switching
            # Add the thumbnail to the grid layout in the scroll area
            row = len(self.uploaded_files) - 1
            self.scroll_layout.addWidget(thumbnail_button, row, 0)

        except Exception as e:
            print(f"Error creating thumbnail for file: {file_path}\n{e}")

    def select_file_by_thumbnail(self, file_path):
        """Handle file selection by clicking on a thumbnail."""
        try:
            self.set_active_file(file_path)  # Ensure we load the selected file
        except Exception as e:
            print(f"Error selecting file from thumbnail: {e}")

    def set_active_file(self, file_path):
        """Set the clicked file as the active file for viewing."""
        try:
            self.current_file = file_path
            nii_file = self.uploaded_files[os.path.basename(file_path)]
            self.img_data = nii_file.get_fdata()
            self.slice_idx = self.img_data.shape[2] // 2
            self.time_idx = 0  # Reset time index when switching files
            self.update_image()

            # Update window title with file name
            self.setWindowTitle(f'NIfTI Viewer - {os.path.basename(file_path)}')

            # Switch between 3D and CINE modes
            self.switch_mode(nii_file)
        except Exception as e:
            print(f"Error activating file: {file_path}\n{e}")

    def wheelEvent(self, event):
        """Handle mouse wheel events for scrolling through frames or slices."""
        angle = event.angleDelta().y()  # Get the amount of scroll
        delta = 1 if angle > 0 else -1  # Determine direction of scroll

        if len(self.img_data.shape) == 4 and self.img_data.shape[3] > 1:
            # CINE mode (scroll through frames)
            self.time_idx = (self.time_idx + delta) % self.img_data.shape[3]
            self.update_image()
        else:
            # 3D mode (scroll through slices)
            self.slice_idx = np.clip(self.slice_idx + delta, 0, self.img_data.shape[2] - 1)
            self.slice_slider.setValue(self.slice_idx)  # Update the slider
            self.update_image()

    def update_image(self):
        """Update the main canvas with the current slice or frame."""
        self.ax.clear()
        try:
            if len(self.img_data.shape) == 4 and self.img_data.shape[3] == 1:
                slice_data = self.img_data[:, :, self.slice_idx, 0]  # 3D image disguised as 4D
                self.frame_slice_label.setText(f"Slice {self.slice_idx}")
            elif len(self.img_data.shape) == 4:
                slice_data = self.img_data[:, :, self.slice_idx, self.time_idx]
                self.frame_slice_label.setText(f"Frame {self.time_idx}")
            else:
                slice_data = self.img_data[:, :, self.slice_idx]
                self.frame_slice_label.setText(f"Slice {self.slice_idx}")

            self.ax.imshow(slice_data, cmap='gray')
            self.ax.set_title(f"Slice {self.slice_idx} (Time {self.time_idx})")
            self.canvas.draw()
        except Exception as e:
            print(f"Error updating image: {e}")

    def switch_mode(self, nii_file):
        """Switch between CINE playback mode and 3D slice scrolling mode."""
        img_shape = nii_file.get_fdata().shape

        # Set visibility based on the type of file (3D or 4D)
        if len(img_shape) == 4 and img_shape[3] > 1:
            # Enable CINE mode for 4D images
            self.play_button.setVisible(True)
            self.stop_button.setVisible(True)
            self.speed_slider.setVisible(True)
            self.playback_speed_label.setVisible(True)
            self.slice_slider.setVisible(False)
            self.slice_depth_label.setVisible(False)
            self.frame_slice_label.setText(f"Frame {self.time_idx}")
            self.frame_slice_label.setVisible(True)
            self.timer.start(self.playback_speed)  # Start CINE playback

        else:  # This is a 3D image (even if shape[3] == 1)
            self.play_button.setVisible(False)
            self.stop_button.setVisible(False)
            self.speed_slider.setVisible(False)
            self.playback_speed_label.setVisible(False)
            self.slice_slider.setVisible(True)
            self.slice_depth_label.setVisible(True)
            self.frame_slice_label.setText(f"Slice {self.slice_idx}")
            self.frame_slice_label.setVisible(True)
            self.slice_slider.setMaximum(img_shape[2] - 1)  # Set max slices based on 3D depth
            self.slice_slider.setValue(self.slice_idx)
            self.slice_slider.valueChanged.connect(self.update_slice)

        # Explicitly stop playback mode if switching from 4D to 3D
        self.playing = False
        self.timer.stop()

    def update_slice(self, value):
        """Update slice index based on the slice scroller."""
        self.slice_idx = value
        self.frame_slice_label.setText(f"Slice {self.slice_idx}")  # Update slice indicator
        self.update_image()

    def toggle_play(self):
        """Toggle between play and stop for CINE mode."""
        if not self.playing:
            self.playing = True
            self.timer.start(self.playback_speed)
        else:
            self.playing = False
            self.timer.stop()

    def stop_playback(self):
        """Stop CINE playback."""
        self.playing = False
        self.timer.stop()

    def next_frame(self):
        """Go to the next frame in CINE mode."""
        if len(self.img_data.shape) == 4:
            self.time_idx = (self.time_idx + 1) % self.img_data.shape[3]
            self.frame_slice_label.setText(f"Frame {self.time_idx}")  # Update frame indicator
        self.update_image()

    def adjust_speed(self, value):
        """Adjust the playback speed for CINE mode."""
        self.playback_speed = value
        self.playback_speed_label.setText(
            f"Playback Speed: {self.playback_speed} ms")  # Update label with current speed
        if self.playing:
            self.timer.start(self.playback_speed)

    def toggle_dark_mode(self, state):
        """Toggle between dark mode and light mode."""
        self.dark_mode_enabled = state == Qt.Checked
        if self.dark_mode_enabled:
            self.apply_dark_mode()
        else:
            self.apply_light_mode()

    def apply_dark_mode(self):
        """Apply dark mode styles."""
        self.ax.set_facecolor('black')
        self.figure.patch.set_facecolor('black')
        self.ax.title.set_color('white')
        self.setStyleSheet("background-color: black; color: white;")

    def apply_light_mode(self):
        """Apply light mode styles."""
        self.ax.set_facecolor('white')
        self.figure.patch.set_facecolor('white')
        self.ax.title.set_color('black')
        self.setStyleSheet("background-color: white; color: black;")

    def upload_file(self):
        """Allow users to upload either NIfTI or DICOM files."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "",
                                                   "All Files (*);;NIfTI Files (*.nii *.nii.gz);;DICOM Files (*.dcm *.DCM)")

        if file_path:
            nifti_file_path = handle_file_upload(file_path)
            if nifti_file_path:
                # Load the NIfTI file after handling
                self.load_nifti_file(nifti_file_path)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Initial file (optional)
    initial_file = None

    viewer = NiftiViewer(initial_file)
    viewer.setWindowTitle('NIfTI Viewer with Thumbnails')
    viewer.show()

    sys.exit(app.exec_())
