# tardis.py
import sys
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QVBoxLayout, QPushButton, QSlider, QWidget, QLabel,
    QHBoxLayout, QCheckBox, QScrollArea, QGridLayout, QFileDialog, QSizePolicy,
    QDesktopWidget, QMenuBar, QAction, QMessageBox, QDialog, QLineEdit, QSplitter, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QMimeData, QSize
from PyQt5.QtGui import QIcon, QPixmap, QImage, QDrag
import os
from app_utils import handle_file_upload, extract_slice, clean_nifti_dir
from functools import partial

from history_stack import HistoryStack

# Import modification functions
from registration_modifications import affine_registration, non_rigid_registration
from filtering_modifications import apply_gaussian_filter, apply_median_filter, apply_non_local_means

# Import threading for background processing
class ModificationThread(QThread):
    modification_complete = pyqtSignal(object, object)  # Emits modified_data and new_affine

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.modification_complete.emit(*result)
        except Exception as e:
            self.modification_complete.emit(None, None)
            print(f"Modification failed: {e}")


class ResamplingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resampling Options")
        self.setModal(True)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()  # Use a local variable 'layout'
        self.setLayout(layout)

        # Image display area
        self.image_layout = QHBoxLayout()

        # Set minimum sizes for better scaling
        self.original_label = QLabel("Original")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(200, 200)
        self.original_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.modified_label = QLabel("Modified")
        self.modified_label.setAlignment(Qt.AlignCenter)
        self.modified_label.setMinimumSize(200, 200)
        self.modified_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.image_layout.addWidget(self.original_label)
        self.image_layout.addWidget(self.modified_label)
        self.layout.addLayout(self.image_layout)

        # Control buttons
        self.button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save As")
        self.reject_button = QPushButton("Reject")
        self.close_button = QPushButton("Close")
        self.save_button.clicked.connect(self.save_as)
        self.reject_button.clicked.connect(self.reject_modification)
        self.close_button.clicked.connect(self.close_preview)
        self.button_layout.addWidget(self.save_button)
        self.button_layout.addWidget(self.reject_button)
        self.button_layout.addWidget(self.close_button)
        self.layout.addLayout(self.button_layout)

    def apply(self):
        try:
            factor = float(self.factor_input.text())
            if factor <= 0:
                raise ValueError("Factor must be positive.")
            self.factor = factor
            self.accept()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Please enter a valid resampling factor.\nError: {e}")


class IntensityNormalizationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Intensity Normalization Options")
        self.setModal(True)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Minimum Intensity Input
        min_layout = QHBoxLayout()
        min_label = QLabel("Minimum Intensity:")
        self.min_input = QLineEdit()
        self.min_input.setPlaceholderText("e.g., 0")
        min_layout.addWidget(min_label)
        min_layout.addWidget(self.min_input)
        layout.addLayout(min_layout)

        # Maximum Intensity Input
        max_layout = QHBoxLayout()
        max_label = QLabel("Maximum Intensity:")
        self.max_input = QLineEdit()
        self.max_input.setPlaceholderText("e.g., 255")
        max_layout.addWidget(max_label)
        max_layout.addWidget(self.max_input)
        layout.addLayout(max_layout)

        # Buttons
        buttons_layout = QHBoxLayout()
        apply_button = QPushButton("Apply")
        cancel_button = QPushButton("Cancel")
        apply_button.clicked.connect(self.apply)
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(apply_button)
        buttons_layout.addWidget(cancel_button)
        layout.addLayout(buttons_layout)

        self.setLayout(layout)

    def apply(self):
        try:
            min_val = float(self.min_input.text())
            max_val = float(self.max_input.text())
            if min_val >= max_val:
                raise ValueError("Minimum must be less than maximum.")
            self.min_val = min_val
            self.max_val = max_val
            self.accept()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Please enter valid intensity values.\nError: {e}")

class RegistrationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Registration")
        self.setModal(True)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Registration Type Selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Registration Type:")
        self.affine_radio = QCheckBox("Affine")
        self.non_rigid_radio = QCheckBox("Non-Rigid")
        self.affine_radio.setChecked(True)
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.affine_radio)
        type_layout.addWidget(self.non_rigid_radio)
        layout.addLayout(type_layout)

        # Reference Image Selection
        ref_layout = QHBoxLayout()
        ref_label = QLabel("Reference Image:")
        self.ref_input = QLineEdit()
        self.ref_browse = QPushButton("Browse")
        self.ref_browse.clicked.connect(self.browse_reference)
        ref_layout.addWidget(ref_label)
        ref_layout.addWidget(self.ref_input)
        ref_layout.addWidget(self.ref_browse)
        layout.addLayout(ref_layout)

        # Buttons
        buttons_layout = QHBoxLayout()
        apply_button = QPushButton("Apply")
        cancel_button = QPushButton("Cancel")
        apply_button.clicked.connect(self.apply)
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(apply_button)
        buttons_layout.addWidget(cancel_button)
        layout.addLayout(buttons_layout)

        self.setLayout(layout)

    def browse_reference(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Reference Image", "",
                                                   "NIfTI Files (*.nii *.nii.gz)")
        if file_path:
            self.ref_input.setText(file_path)

    def apply(self):
        registration_type = None
        if self.affine_radio.isChecked() and not self.non_rigid_radio.isChecked():
            registration_type = "Affine"
        elif self.non_rigid_radio.isChecked() and not self.affine_radio.isChecked():
            registration_type = "Non-Rigid"
        elif self.affine_radio.isChecked() and self.non_rigid_radio.isChecked():
            QMessageBox.warning(self, "Selection Error", "Please select only one registration type.")
            return
        else:
            QMessageBox.warning(self, "Selection Error", "Please select a registration type.")
            return

        reference_file = self.ref_input.text()
        if not reference_file or not os.path.isfile(reference_file):
            QMessageBox.warning(self, "Reference Image", "Please select a valid reference image.")
            return

        self.registration_type = registration_type
        self.reference_file = reference_file
        self.accept()

class FilteringDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filtering Options")
        self.setModal(True)
        self.selected_filters = None  # Initialize selected_filters
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Filter Type Selection using QRadioButton
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter Type:")

        self.gaussian_radio = QRadioButton("Gaussian")
        self.median_radio = QRadioButton("Median")
        self.nlm_radio = QRadioButton("Non-Local Means")
        self.gaussian_radio.setChecked(True)

        # Group the radio buttons
        self.filter_group = QButtonGroup()
        self.filter_group.addButton(self.gaussian_radio)
        self.filter_group.addButton(self.median_radio)
        self.filter_group.addButton(self.nlm_radio)
        self.filter_group.buttonClicked.connect(self.update_parameters)

        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.gaussian_radio)
        filter_layout.addWidget(self.median_radio)
        filter_layout.addWidget(self.nlm_radio)
        layout.addLayout(filter_layout)

        # Parameter Inputs
        self.param_layout = QVBoxLayout()

        # Gaussian Parameters
        self.gaussian_param_layout = QHBoxLayout()
        self.gaussian_label = QLabel("Sigma:")
        self.gaussian_input = QLineEdit()
        self.gaussian_input.setPlaceholderText("e.g., 1.0")
        self.gaussian_param_layout.addWidget(self.gaussian_label)
        self.gaussian_param_layout.addWidget(self.gaussian_input)
        self.gaussian_param_widget = QWidget()
        self.gaussian_param_widget.setLayout(self.gaussian_param_layout)
        self.param_layout.addWidget(self.gaussian_param_widget)

        # Median Parameters
        self.median_param_layout = QHBoxLayout()
        self.median_label = QLabel("Kernel Size:")
        self.median_input = QLineEdit()
        self.median_input.setPlaceholderText("e.g., 3")
        self.median_param_layout.addWidget(self.median_label)
        self.median_param_layout.addWidget(self.median_input)
        self.median_param_widget = QWidget()
        self.median_param_widget.setLayout(self.median_param_layout)
        self.param_layout.addWidget(self.median_param_widget)
        self.median_param_widget.setVisible(False)

        # Non-Local Means Parameters
        self.nlm_param_layout = QHBoxLayout()
        self.nlm_patch_label = QLabel("Patch Size:")
        self.nlm_patch_input = QLineEdit()
        self.nlm_patch_input.setPlaceholderText("e.g., 5")
        self.nlm_distance_label = QLabel("Patch Distance:")
        self.nlm_distance_input = QLineEdit()
        self.nlm_distance_input.setPlaceholderText("e.g., 6")
        self.nlm_h_label = QLabel("H Parameter:")
        self.nlm_h_input = QLineEdit()
        self.nlm_h_input.setPlaceholderText("e.g., 0.1")
        self.nlm_param_layout.addWidget(self.nlm_patch_label)
        self.nlm_param_layout.addWidget(self.nlm_patch_input)
        self.nlm_param_layout.addWidget(self.nlm_distance_label)
        self.nlm_param_layout.addWidget(self.nlm_distance_input)
        self.nlm_param_layout.addWidget(self.nlm_h_label)
        self.nlm_param_layout.addWidget(self.nlm_h_input)
        self.nlm_param_widget = QWidget()
        self.nlm_param_widget.setLayout(self.nlm_param_layout)
        self.param_layout.addWidget(self.nlm_param_widget)
        self.nlm_param_widget.setVisible(False)

        layout.addLayout(self.param_layout)

        # Buttons
        buttons_layout = QHBoxLayout()
        apply_button = QPushButton("Apply")
        cancel_button = QPushButton("Cancel")
        apply_button.clicked.connect(self.apply)
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(apply_button)
        buttons_layout.addWidget(cancel_button)
        layout.addLayout(buttons_layout)

        self.setLayout(layout)

    def update_parameters(self):
        self.gaussian_param_widget.setVisible(self.gaussian_radio.isChecked())
        self.median_param_widget.setVisible(self.median_radio.isChecked())
        self.nlm_param_widget.setVisible(self.nlm_radio.isChecked())

    def apply(self):
        try:
            # Determine which filter is selected and store the parameters
            if self.gaussian_radio.isChecked():
                sigma = float(self.gaussian_input.text())
                self.selected_filters = {'type': 'gaussian', 'sigma': sigma}
            elif self.median_radio.isChecked():
                kernel_size = int(self.median_input.text())
                self.selected_filters = {'type': 'median', 'kernel_size': kernel_size}
            elif self.nlm_radio.isChecked():
                patch_size = int(self.nlm_patch_input.text())
                patch_distance = int(self.nlm_distance_input.text())
                h_param = float(self.nlm_h_input.text())
                self.selected_filters = {
                    'type': 'nlm',
                    'patch_size': patch_size,
                    'patch_distance': patch_distance,
                    'h_param': h_param
                }
            self.accept()
        except ValueError as ve:
            QMessageBox.critical(self, "Input Error", f"Invalid input: {ve}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")

# Similarly, create RegistrationDialog, DenoisingDialog, and ROTrackingDialog
# Dialog Classes (ResamplingDialog, IntensityNormalizationDialog, RegistrationDialog, FilteringDialog)

class DraggableThumbnail(QPushButton):
    def __init__(self, file_path, icon, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.setIcon(icon)
        self.setIconSize(icon.availableSizes()[0])
        self.setFixedSize(icon.availableSizes()[0] + QSize(20, 20))  # Adjust size as needed

    def mouseMoveEvent(self, event):
        if event.buttons() != Qt.LeftButton:
            return

        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(self.file_path)  # Set file path as text in MIME data
        drag.setMimeData(mime_data)

        # Optional: Set drag pixmap
        if not self.icon().isNull():
            drag.setPixmap(self.icon().pixmap(self.iconSize()))

        print(f"Dragging file: {self.file_path}")  # Debugging print
        drag.exec_(Qt.CopyAction | Qt.MoveAction)


class ComparisonWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.setAcceptDrops(True)  # Accept drops
        self.slice_idx = 0  # Initialize the Z-axis slice index
        self.time_idx = 0  # Initialize the time index for 4D images
        self.is_4d = False  # Track if the image is 4D

    def init_ui(self):
        layout = QVBoxLayout()

        # Image Label
        self.image_label = QLabel("Drop a thumbnail here to compare")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #000000; color: white;") # grey 2E2E2E
        self.image_label.setScaledContents(False)
        layout.addWidget(self.image_label)

        # Close Button
        close_button = QPushButton("Close Comparison")
        close_button.clicked.connect(self.close_comparison)
        layout.addWidget(close_button)

        # Slice slider for scrolling through slices (Z-axis)
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self.update_slice)
        self.slice_slider.setVisible(False)  # Initially hidden
        layout.addWidget(self.slice_slider)

        # Time slider for scrolling through frames (T-axis) in 4D images
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.valueChanged.connect(self.update_time)
        self.time_slider.setVisible(False)  # Initially hidden
        layout.addWidget(self.time_slider)

        self.setLayout(layout)
        self.hide()  # Hidden initially

        # Set initial style
        self.default_style = """
            QWidget {
                background-color: #2E2E2E;
                border: 2px dashed #555555;
            }
        """
        self.highlight_style = """
            QWidget {
                background-color: #3E3E3E;
                border: 2px dashed #FF0000;
            }
        """
        self.setStyleSheet(self.default_style)

    def set_image(self, image_data):
        """Set and display the image in the ComparisonWidget."""
        try:
            self.image_data = image_data
            self.is_4d = image_data.ndim == 4  # Check if the image is 4D

            # Remove old sliders if they exist
            if self.slice_slider is not None:
                self.slice_slider.deleteLater()  # Ensure old slider is deleted
                self.slice_slider = None  # Reset to None after deletion
            if self.time_slider is not None:
                self.time_slider.deleteLater()  # Ensure old slider is deleted
                self.time_slider = None  # Reset to None after deletion

            # Recreate the slice slider (Z-axis)
            self.slice_slider = QSlider(Qt.Horizontal)
            self.slice_slider.setMaximum(self.image_data.shape[2] - 1)
            self.slice_slider.valueChanged.connect(self.update_slice)
            self.layout().addWidget(self.slice_slider)

            # Recreate the time slider (T-axis) only for 4D images
            if self.is_4d:
                self.time_slider = QSlider(Qt.Horizontal)
                self.time_slider.setMaximum(self.image_data.shape[3] - 1)
                self.time_slider.valueChanged.connect(self.update_time)
                self.layout().addWidget(self.time_slider)
                self.time_slider.setVisible(True)
            else:
                self.time_slider = None

            # Display the initial slice and time frame
            self.update_display()

        except Exception as e:
            print(f"Failed to set comparison image: {e}")
            QMessageBox.critical(self, "Error", f"Failed to set comparison image:\n{e}")

    def reconnect_signals(self):
        """Reconnect signals for sliders."""
        self.slice_slider.valueChanged.connect(self.update_slice)
        if self.is_4d:
            self.time_slider.valueChanged.connect(self.update_time)

    def update_slice(self, value):
        """Update the Z-axis slice index."""
        self.slice_idx = value
        self.update_display()

    def update_time(self, value):
        """Update the time index for 4D images."""
        self.time_idx = value
        self.update_display()

    def update_display(self):
        """Update the image display based on the current slice and time index."""
        try:
            if self.image_data is None:
                return  # No image data to display

            if self.is_4d:
                # For 4D images, use both the slice and time index
                if self.slice_slider and self.time_slider:
                    slice_data = self.image_data[:, :, self.slice_slider.value(), self.time_slider.value()]
            else:
                # For 3D images, use only the slice index
                if self.slice_slider:
                    slice_data = self.image_data[:, :, self.slice_slider.value()]

            # Normalize the data for display
            normalized = ((slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255).astype(
                np.uint8)
            height, width = normalized.shape

            # Ensure the data is contiguous in memory
            if not normalized.flags['C_CONTIGUOUS']:
                normalized = np.ascontiguousarray(normalized)

            # Convert numpy array to QImage
            q_image = QImage(normalized.data, width, height, width, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)

            # Set the pixmap to the label
            self.image_label.setPixmap(pixmap)
            self.show()  # Ensure the widget is visible

        except Exception as e:
            print(f"Error updating display: {e}")
            QMessageBox.critical(self, "Error", f"Error updating display:\n{e}")

    def close_comparison(self):
        """Close the comparison area and safely clean up resources."""
        try:
            self.hide()  # Hide the widget from view

            # Clear image data
            self.image_data = None

            # Clean up sliders properly by deleting them, but only if they exist
            if self.slice_slider is not None:
                self.slice_slider.deleteLater()
                self.slice_slider = None
            if self.time_slider is not None:
                self.time_slider.deleteLater()
                self.time_slider = None

            # Notify the parent to resize the main image
            parent_window = self.window()
            if hasattr(parent_window, 'resize_main_image'):
                parent_window.resize_main_image()

        except Exception as e:
            print(f"Error while closing comparison: {e}")
            QMessageBox.critical(self, "Error", f"Failed to close comparison:\n{e}")


class NiftiViewer(QMainWindow):
    def __init__(self, initial_file=None):
        super().__init__()
        self.setAcceptDrops(True)  # Accept drops

        # Initialize variables
        self.uploaded_files = {}
        self.thumbnail_containers = {}
        self.current_file = None
        self.img_data = None
        self.img_affine = None
        self.slice_idx = 0
        self.time_idx = 0
        self.playing = False
        self.dark_mode_enabled = True
        self.playback_speed = 100

        # Initialize UI components
        self.init_ui()

        # Initialize history stack
        self.history = HistoryStack()

        # Timer for frame updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        # Get monitor resolution and resize accordingly
        self.screen_resolution = QApplication.desktop().screenGeometry()
        self.screen_width = self.screen_resolution.width()
        self.screen_height = self.screen_resolution.height()

        self.resize(int(self.screen_width * 0.8), int(self.screen_height * 0.8))  # Resize window to 80% of screen size

        # Load initial file if provided
        if initial_file:
            self.load_nifti_file(initial_file)

    def closeEvent(self, event):
        """Handle the closing of the widget and ensure proper cleanup."""
        try:
            # Stop any background threads or timers
            if hasattr(self, 'background_thread') and self.background_thread.isRunning():
                self.background_thread.quit()
                self.background_thread.wait()

            # Call parent class closeEvent
            super().closeEvent(event)

        except Exception as e:
            print(f"Error while closing widget: {e}")
            event.ignore()  # Ignore the close event if an error occurs

    def init_ui(self):
        # Main widget
        widget = QWidget()
        self.setCentralWidget(widget)

        # Main horizontal layout using QSplitter
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout = QHBoxLayout()
        self.main_layout.addWidget(self.splitter)
        widget.setLayout(self.main_layout)

        # Left sidebar for thumbnails (scrollable)
        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedWidth(250)  # Fixed width for thumbnails
        self.splitter.addWidget(self.scroll_area)
        self.thumbnail_widgets = {}  # Dictionary to store thumbnails

        # Central area: Image display and controls
        central_widget = QWidget()
        self.right_layout = QVBoxLayout()  # Define right_layout for central content
        central_widget.setLayout(self.right_layout)

        # Canvas for displaying the image
        self.figure, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_anchor('C')  # Center the image initially
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        self.right_layout.addWidget(self.canvas)

        # Buttons and sliders for controls
        self.add_controls()

        # Button to upload additional files
        self.upload_button = QPushButton("Upload New File")
        self.upload_button.clicked.connect(self.upload_file)
        self.right_layout.addWidget(self.upload_button)

        # Add central widget to splitter
        self.splitter.addWidget(central_widget)

        # Right sidebar for ComparisonWidget
        self.comparison_widget = ComparisonWidget(self)
        self.comparison_widget.setFixedWidth(400)
        self.comparison_widget.show()  # Ensure it's shown and ready to accept drops
        self.splitter.addWidget(self.comparison_widget)

        # Set initial sizes for the splitter sections (adjusted for three panels)
        self.splitter.setSizes([250, 1050, 0])  # Start without comparison

        # Apply default dark mode
        self.apply_dark_mode()

        # Initialize the menu bar
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)

        # Add "Inspect" and "Modifications" menus
        self.add_inspect_menu()
        self.add_modifications_menu()
        self.add_edit_menu()

    def add_inspect_menu(self):
        inspect_menu = self.menu_bar.addMenu("Inspect")

        # Add "Show File Info" action
        show_file_info_action = QAction("Show File Info", self)
        show_file_info_action.triggered.connect(self.show_file_info)
        inspect_menu.addAction(show_file_info_action)

    def add_modifications_menu(self):
        modifications_menu = self.menu_bar.addMenu("Modifications")

        # List of modification actions
        modifications = [
            ("Resampling", self.open_resampling_dialog),
            ("Intensity Normalization", self.open_intensity_normalization_dialog),
            ("Registration", self.open_registration_dialog),
            ("Denoising", self.open_denoising_dialog),
            ("ROI Tracking", self.open_roi_tracking_dialog),
        ]

        for name, callback in modifications:
            action = QAction(name, self)
            action.triggered.connect(callback)
            modifications_menu.addAction(action)

    def add_edit_menu(self):
        edit_menu = self.menu_bar.addMenu("Edit")

        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.triggered.connect(self.undo)
        self.undo_action.setEnabled(False)  # Initially disabled
        edit_menu.addAction(self.undo_action)

        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcut("Ctrl+Y")
        self.redo_action.triggered.connect(self.redo)
        self.redo_action.setEnabled(False)  # Initially disabled
        edit_menu.addAction(self.redo_action)

    def undo(self):
        state = self.history.undo()
        if state is not None:
            self.img_data = state
            self.update_image()
            self.update_undo_redo_actions()
        else:
            QMessageBox.information(self, "Undo", "Nothing to undo.")

    def redo(self):
        state = self.history.redo()
        if state is not None:
            self.img_data = state
            self.update_image()
            self.update_undo_redo_actions()
        else:
            QMessageBox.information(self, "Redo", "Nothing to redo.")

    def update_undo_redo_actions(self):
        """Enable or disable Undo/Redo actions based on history stack."""
        self.undo_action.setEnabled(self.history.can_undo())
        self.redo_action.setEnabled(self.history.can_redo())

    def show_file_info(self):
        if not self.current_file:
            QMessageBox.warning(self, "No File Loaded", "Please load a file to view its information.")
            return

        try:
            nii_file = nib.load(self.current_file)
            img_shape = nii_file.shape
            header = nii_file.header
            img_spacing = header.get_zooms()

            # Prepare the information string
            info_text = f"Shape: {img_shape}\nSpacing: {img_spacing}"

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to retrieve file information:\n{e}")
            return

        # Create and display the information dialog
        info_dialog = QMessageBox(self)
        info_dialog.setWindowTitle("File Information")
        info_dialog.setText(info_text)
        info_dialog.setIcon(QMessageBox.Information)
        info_dialog.setStandardButtons(QMessageBox.Ok)
        info_dialog.exec_()

    def dragEnterEvent(self, event):
        """Handle drag enter event."""
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Handle drop event."""
        file_path = event.mimeData().text()
        if os.path.isfile(file_path):
            self.load_comparison_file(file_path)
            event.acceptProposedAction()
        else:
            QMessageBox.warning(self, "Drop Error", "The dropped item is not a valid file.")
            event.ignore()

    def load_comparison_file(self, file_path):
        """Load and display the comparison file."""
        try:
            print(f"Attempting to load comparison file: {file_path}")  # Debugging

            # Load the NIfTI file using nibabel
            nii_file = nib.load(file_path)
            comparison_data = nii_file.get_fdata()

            # Display the full image in the ComparisonWidget (3D or 4D)
            self.comparison_widget.set_image(comparison_data)

            # Resize the main image display to accommodate the comparison
            self.resize_main_image()

        except Exception as e:
            print(f"Error loading comparison file: {file_path}\n{e}")  # Debugging the error
            QMessageBox.critical(self, "Error", f"Failed to load comparison file:\n{e}")

    def resize_main_image(self):
        """Resize the main image display to accommodate the comparison."""
        if self.comparison_widget.isVisible():
            # Adjust the splitter sizes to give space to both main and comparison images
            self.splitter.setSizes([250, 800, 400])
        else:
            # Restore the splitter sizes when comparison is closed
            self.splitter.setSizes([250, 1050, 0])

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
            nii_file = nib.load(file_path)
            self.img_data = nii_file.get_fdata()
            self.img_affine = nii_file.affine  # Store affine matrix
            self.slice_idx = self.img_data.shape[2] // 2  # Default middle slice
            self.time_idx = 0  # Reset time index when switching files
            self.update_image()

            # Update window title with file name
            self.setWindowTitle(f'TARDIS - {os.path.basename(file_path)}')

            # Switch between 3D and CINE modes based on file type
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
        if len(self.img_data.shape) == 4 and self.img_data.shape[3] > 1:
            self.time_idx = (self.time_idx + 1) % self.img_data.shape[3]
            self.frame_slice_label.setText(f"Frame {self.time_idx}")
            self.update_image()

    def adjust_speed(self, value):
        """Adjust the playback speed for CINE mode."""
        self.playback_speed = value
        self.playback_speed_label.setText(f"Playback Speed: {self.playback_speed} ms")
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

    def show_preview(self, original, modified):
        """Removed PreviewManager usage."""
        pass  # No longer needed

    def load_nifti_file(self, file_path):
        """Load the NIfTI file and update the main image display."""
        try:
            nii_file = nib.load(file_path)
            self.add_thumbnail(file_path)  # Now, let add_thumbnail handle the addition to uploaded_files

            # Load the first file by default if no file is currently active
            if not self.current_file:
                self.set_active_file(file_path)

            # Use SimpleITK to read temporal resolution from the NIfTI header
            itk_img = sitk.ReadImage(file_path)
            img_spacing = itk_img.GetSpacing()

            # If it's a CINE scan (4D image), set the playback speed to the temporal resolution
            if len(self.img_data.shape) == 4 and self.img_data.shape[3] > 1:
                temporal_resolution = img_spacing[3] * 1000  # Convert seconds to milliseconds
                self.playback_speed = max(10, int(temporal_resolution))  # Ensure minimum speed
                self.playback_speed_label.setText(f"Playback Speed: {self.playback_speed} ms")
                self.speed_slider.setValue(self.playback_speed)

            # Update window title with file name
            self.setWindowTitle(f'TARDIS - {os.path.basename(file_path)}')

        except Exception as e:
            print(f"Error loading file: {file_path}\n{e}")
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")

    def add_thumbnail(self, file_path):
        """Create a thumbnail for the file and add it to the sidebar with a delete button."""
        try:
            filename = os.path.basename(file_path)

            # Only add the file if it hasn't already been added
            if filename in self.uploaded_files:
                self.delete_file(file_path)  # Clean up if the file was previously added

            # Now, add the file to the dictionary of uploaded files after the check
            self.uploaded_files[filename] = file_path
            nii_file = nib.load(file_path)
            middle_slice = extract_slice(nii_file)

            # Create thumbnail image
            fig, ax = plt.subplots(figsize=(1.7, 1.7))
            ax.imshow(middle_slice, cmap='gray')
            ax.axis('off')
            fig.patch.set_facecolor("black")
            plt.tight_layout()

            # Convert the plot to a canvas and use it as a thumbnail
            canvas = FigureCanvas(fig)
            canvas.draw()

            # Convert canvas to QPixmap and then to QIcon
            thumbnail_pixmap = QPixmap(canvas.grab())
            thumbnail_icon = QIcon(thumbnail_pixmap)

            # Create a draggable thumbnail button
            thumbnail_button = DraggableThumbnail(file_path, thumbnail_icon)
            thumbnail_button.clicked.connect(partial(self.select_file_by_thumbnail, file_path))

            # Create delete button
            delete_button = QPushButton("X")  # Simple 'X' button for delete
            delete_button.setFixedSize(20, 20)  # Set a small size for the delete button
            delete_button.clicked.connect(partial(self.delete_file, file_path))  # Connect to delete function

            # Create a layout for the thumbnail and delete button
            thumbnail_layout = QHBoxLayout()
            thumbnail_layout.addWidget(thumbnail_button)
            thumbnail_layout.addWidget(delete_button)

            # Create a container widget to hold both buttons
            thumbnail_container = QWidget()
            thumbnail_container.setLayout(thumbnail_layout)

            # Add the thumbnail container directly to the scroll layout
            self.scroll_layout.addWidget(thumbnail_container)
            self.thumbnail_containers[filename] = thumbnail_container

        except Exception as e:
            print(f"Error creating thumbnail for file: {file_path}\n{e}")

    def start_drag(self, event, file_path):
        """Initiate drag event for thumbnails."""
        drag = QDrag(self)
        mime_data = QMimeData()
        url = QUrl.fromLocalFile(file_path)
        mime_data.setUrls([url])
        drag.setMimeData(mime_data)

        # Optional: Set drag icon
        thumbnail_button = self.sender()
        if thumbnail_button.icon():
            drag.setPixmap(thumbnail_button.icon().pixmap(thumbnail_button.iconSize()))

        drag.exec_(Qt.CopyAction | Qt.MoveAction)

    def delete_file(self, file_path):
        """Delete the selected file and remove it from the view without deleting from disk."""
        try:
            filename = os.path.basename(file_path)

            if filename in self.uploaded_files:
                del self.uploaded_files[filename]
            else:
                pass

            # Remove the thumbnail container
            if filename in self.thumbnail_containers:
                container = self.thumbnail_containers.pop(filename)
                self.scroll_layout.removeWidget(container)
                container.setParent(None)
                container.deleteLater()

            # Clear current_file if it matches the deleted file
            if self.current_file == file_path:
                self.current_file = None
                self.img_data = None
                self.img_affine = None
                self.ax.clear()
                self.canvas.draw()
                self.setWindowTitle('TARDIS - No File Selected')

            # If the deleted file was in comparison, close comparison
            if self.comparison_widget.isVisible() and self.comparison_widget.image_data is not None:
                comparison_file = None
                # Find which file is in comparison by matching image data
                for fname, fpath in self.uploaded_files.items():
                    try:
                        nii = nib.load(fpath)
                        data = extract_slice(nii)
                        if np.array_equal(data, self.comparison_widget.image_data):
                            comparison_file = fpath
                            break
                    except:
                        continue
                if comparison_file == file_path:
                    self.comparison_widget.close_comparison()

        except Exception as e:
            print(f"Error deleting file: {e}")

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
            self.ax.set_title(self.frame_slice_label.text())
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
        if len(self.img_data.shape) == 4 and self.img_data.shape[3] > 1:
            self.time_idx = (self.time_idx + 1) % self.img_data.shape[3]
            self.frame_slice_label.setText(f"Frame {self.time_idx}")
            self.update_image()

    def adjust_speed(self, value):
        """Adjust the playback speed for CINE mode."""
        self.playback_speed = value
        self.playback_speed_label.setText(f"Playback Speed: {self.playback_speed} ms")
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
        self.slice_slider.setMinimum(0)
        self.slice_slider.valueChanged.connect(self.update_slice)
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

    def apply_modification(self, modified_data, new_affine=None):
        """Apply the modification and update history."""
        if new_affine is not None:
            self.img_affine = new_affine  # Update affine matrix if provided
        self.history.push(self.img_data.copy())
        self.img_data = modified_data
        self.update_image()
        self.update_undo_redo_actions()

    def open_resampling_dialog(self):
        dialog = ResamplingDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            factor = dialog.factor
            self.apply_resampling(factor)

    def open_intensity_normalization_dialog(self):
        dialog = IntensityNormalizationDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            min_val = dialog.min_val
            max_val = dialog.max_val
            self.apply_intensity_normalization(min_val, max_val)

    def open_registration_dialog(self):
        dialog = RegistrationDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            registration_type = dialog.registration_type
            reference_file = dialog.reference_file
            self.perform_registration(registration_type, reference_file)

    def open_denoising_dialog(self):
        dialog = FilteringDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            selected_filters = dialog.selected_filters
            if selected_filters is None:
                QMessageBox.warning(self, "Warning", "No filters selected.")
            else:
                self.apply_filtering(selected_filters)

    def open_roi_tracking_dialog(self):
        # Implement ROTrackingDialog similar to ResamplingDialog
        QMessageBox.information(self, "Info", "ROI Tracking dialog not implemented yet.")

    def apply_resampling(self, factor):
        if not self.current_file:
            QMessageBox.warning(self, "No File Loaded", "Please load a file to apply modifications.")
            return

        try:
            original_data = self.img_data.copy()
            # Start a thread to perform resampling
            thread = ModificationThread(self.resample_algorithm, original_data, factor)
            thread.modification_complete.connect(self.on_modification_complete)
            thread.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply resampling:\n{e}")

    def apply_intensity_normalization(self, min_val, max_val):
        if not self.current_file:
            QMessageBox.warning(self, "No File Loaded", "Please load a file to apply modifications.")
            return

        try:
            original_data = self.img_data.copy()
            modified_data = self.normalize_intensity(original_data, min_val, max_val)
            if modified_data is None:
                raise ValueError("Intensity normalization failed.")
            # Apply the modification directly
            self.apply_modification(modified_data)
            QMessageBox.information(self, "Intensity Normalization", "Intensity normalization applied successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply intensity normalization:\n{e}")

    def perform_registration(self, registration_type, reference_file):
        if not self.current_file:
            QMessageBox.warning(self, "No File Loaded", "Please load a file to apply modifications.")
            return

        try:
            reference_nii = nib.load(reference_file)
            reference_data = reference_nii.get_fdata()
            reference_affine = reference_nii.affine

            original_data = self.img_data.copy()
            original_affine = self.img_affine.copy()

            if registration_type == "Affine":
                func = affine_registration
            elif registration_type == "Non-Rigid":
                func = non_rigid_registration
            else:
                QMessageBox.warning(self, "Registration Type", "Unknown registration type selected.")
                return

            # Start a thread to perform registration
            thread = ModificationThread(func, original_data, original_affine, reference_data, reference_affine)
            thread.modification_complete.connect(self.on_registration_complete)
            thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply registration:\n{e}")

    def on_registration_complete(self, modified_data, new_affine):
        if modified_data is not None:
            self.apply_modification(modified_data, new_affine)
            QMessageBox.information(self, "Registration", "Registration completed successfully.")
        else:
            QMessageBox.critical(self, "Registration Failed", "Registration encountered an error.")

    def apply_filtering(self, selected_filters):
        if not self.current_file:
            QMessageBox.warning(self, "No File Loaded", "Please load a file to apply modifications.")
            return

        try:
            original_data = self.img_data.copy()
            modified_data = original_data.copy()

            filter_name = selected_filters.get('type')

            # Apply each selected filter sequentially
            if filter_name == 'gaussian':
                sigma = selected_filters.get('sigma')
                modified_data = apply_gaussian_filter(modified_data, sigma)
            elif filter_name == 'median':
                kernel_size = selected_filters.get('kernel_size')
                modified_data = apply_median_filter(modified_data, kernel_size)
            elif filter_name == 'nlm':
                patch_size = selected_filters.get('patch_size')
                patch_distance = selected_filters.get('patch_distance')
                h_param = selected_filters.get('h_param')
                modified_data = apply_non_local_means(modified_data, patch_size, patch_distance, h_param)
            else:
                QMessageBox.warning(self, "Filter Error", f"Unknown filter: {filter_name}")
                return

            if modified_data is None:
                raise ValueError("Filtering failed.")

            # Apply the modification directly
            self.apply_modification(modified_data)
            QMessageBox.information(self, "Filtering", "Filtering applied successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply filtering:\n{e}")

    def on_modification_complete(self, modified_data, new_affine=None):
        if modified_data is not None:
            self.apply_modification(modified_data, new_affine)
            QMessageBox.information(self, "Resampling", "Resampling completed successfully.")
        else:
            QMessageBox.critical(self, "Resampling Failed", "Resampling encountered an error.")

    def normalize_intensity(self, img_data, min_val, max_val):
        """Intensity normalization."""
        try:
            img_min = np.min(img_data)
            img_max = np.max(img_data)
            if img_max - img_min == 0:
                raise ValueError("Image has zero intensity range.")
            normalized = (img_data - img_min) / (img_max - img_min)  # Scale to [0,1]
            normalized = normalized * (max_val - min_val) + min_val  # Scale to [min_val, max_val]
            return normalized
        except Exception as e:
            print(f"Intensity normalization failed: {e}")
            return None  # Indicate failure

    def resample_algorithm(self, img_data, factor):
        """Resampling algorithm using Nibabel."""
        try:
            # Create a Nibabel Nifti1Image
            nifti_img = nib.Nifti1Image(img_data, self.img_affine)

            # Define the resampling factor
            new_affine = nifti_img.affine.copy()
            new_affine[:3, :3] = new_affine[:3, :3] / factor  # Adjust spacing

            # Compute new shape
            new_shape = np.ceil(np.array(nifti_img.shape) * factor).astype(int)

            # Perform resampling using Nibabel's processing
            resampled_img = resample_from_to(nifti_img, target_affine=new_affine, target_shape=new_shape,
                                             order=1)  # order=1 for linear
            resampled_data = resampled_img.get_fdata()
            resampled_affine = resampled_img.affine

            return resampled_data, resampled_affine
        except Exception as e:
            print(f"Resampling failed: {e}")
            return None, None  # Indicate failure


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Initial file (optional)
    initial_file = None

    viewer = NiftiViewer(initial_file)
    viewer.setWindowTitle('TARDIS - No File Selected')
    viewer.show()

    sys.exit(app.exec_())
