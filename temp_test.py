# Tardis.py
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
    QDesktopWidget, QMenuBar, QAction, QMessageBox, QDialog, QLineEdit, QSplitter
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap
import os
from app_utils import handle_file_upload, extract_slice, clean_nifti_dir
from preview_manager import PreviewManager
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

# Dialog Classes (ResamplingDialog, IntensityNormalizationDialog, RegistrationDialog, FilteringDialog)
# ... [As provided earlier, no changes needed here] ...
class ResamplingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resampling Options")
        self.setModal(True)
        self.setup_ui()

    def setup_ui(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

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

        self.setLayout(layout)

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

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton,
    QButtonGroup, QLineEdit, QPushButton, QWidget, QMessageBox
)

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


class NiftiViewer(QMainWindow):
    def __init__(self, initial_file=None):
        super().__init__()

        # Dictionary to store all uploaded files (filename -> file_path)
        self.uploaded_files = {}  # filename -> file_path

        self.current_file = None  # Initialize current file
        self.img_data = None
        self.img_affine = None  # Initialize affine matrix

        self.slice_idx = 0
        self.time_idx = 0
        self.playing = False
        self.dark_mode_enabled = True
        self.playback_speed = 100  # Default playback speed, will be updated for CINE

        # Initialize UI
        self.init_ui()

        # Initialize history stack
        self.history = HistoryStack()

        # Add Undo/Redo actions
        self.add_edit_menu()

        # Timer for frame updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        # Get monitor resolution and resize accordingly
        self.screen_resolution = QApplication.desktop().screenGeometry()
        self.screen_width = self.screen_resolution.width()
        self.screen_height = self.screen_resolution.height()

        self.resize(int(self.screen_width * 0.8), int(self.screen_height * 0.8))  # Resize window to 80% of screen size

        # If there's an initial file, load it
        if initial_file:
            self.load_nifti_file(initial_file)

    def closeEvent(self, event):
        try:
            clean_nifti_dir()
        except:
            pass
        event.accept()

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

        # Right sidebar for PreviewManager
        self.preview_manager = PreviewManager(self)
        self.preview_manager.setFixedWidth(400)  # Fixed width for preview
        self.preview_manager.hide()  # Initially hidden
        self.splitter.addWidget(self.preview_manager)

        # Set initial sizes for the splitter sections (ratios)
        self.splitter.setSizes([250, 800, 0])  # Start without preview

        # Apply default dark mode
        self.apply_dark_mode()

        # Initialize the menu bar
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)

        # Add "Inspect" and "Modifications" menus
        self.add_inspect_menu()
        self.add_modifications_menu()

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

    def toggle_preview_panel(self, show):
        """Adjust the splitter to show or hide the preview panel."""
        if show:
            self.preview_manager.show()
            self.splitter.setSizes([250, 600, 400])
        else:
            self.preview_manager.hide()
            self.splitter.setSizes([250, 1000, 0])  # Adjust sizes as needed

    def apply_modification(self, modified_data, new_affine=None):
        """Show preview before applying the modification."""
        if new_affine is not None:
            self.img_affine = new_affine  # Update affine matrix if provided
        self.preview_manager.show_preview(self.img_data, modified_data)

    def accept_modification(self, modified_data):
        """Accept the modification and update history."""
        # Push current state to history before applying
        self.history.push(self.img_data.copy())
        self.img_data = modified_data
        self.update_image()
        self.update_undo_redo_actions()

    def reject_modification(self):
        """Reject the pending modification and hide the preview."""
        self.preview_manager.hide()
        self.toggle_preview_panel(False)
        QMessageBox.information(self, "Modification Rejected", "The modification has been discarded.")

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
            # Push to history and apply
            self.history.push(self.img_data.copy())
            self.img_data = modified_data
            self.update_image()
            self.update_undo_redo_actions()
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
            thread.modification_complete.connect(self.on_registration_complete)  # Ensure connection
            thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply registration:\n{e}")

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

            # Push to history and apply
            self.history.push(self.img_data.copy())
            self.img_data = modified_data
            self.update_image()
            self.update_undo_redo_actions()
            QMessageBox.information(self, "Filtering", "Filtering applied successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply filtering:\n{e}")

    def on_modification_complete(self, modified_data, new_affine):
        if modified_data is not None:
            self.apply_modification(modified_data, new_affine)
            QMessageBox.information(self, "Resampling", "Resampling completed successfully.")
        else:
            QMessageBox.critical(self, "Resampling Failed", "Resampling encountered an error.")

    def apply_modification_from_thread(self, modified_data, new_affine=None):
        """Handle modifications completed by threads."""
        if modified_data is not None:
            self.apply_modification(modified_data, new_affine)
        else:
            QMessageBox.critical(self, "Modification Failed", "The modification could not be completed.")

    def apply_registration_from_thread(self, modified_data, new_affine=None):
        """Handle registration completed by threads."""
        if modified_data is not None:
            self.apply_modification(modified_data, new_affine)
            QMessageBox.information(self, "Registration", "Registration completed successfully.")
        else:
            QMessageBox.critical(self, "Registration Failed", "Registration encountered an error.")

    def resample_algorithm(self, img_data, factor):
        """Resampling algorithm using Nibabel."""
        """
        try:
            # Create a Nibabel Nifti1Image
            nifti_img = nib.Nifti1Image(img_data, self.img_affine)

            # Define the resampling factor
            new_affine = nifti_img.affine.copy()
            new_affine[:3, :3] = new_affine[:3, :3] / factor  # Adjust spacing

            # Compute new shape
            new_shape = np.ceil(np.array(nifti_img.shape) * factor).astype(int)

            # Perform resampling using Nibabel's processing
            resampled_img = nib.processing.resample_from_to(nifti_img, target_affine=new_affine, target_shape=new_shape,
                                                    order=1)  # order=1 for linear
            resampled_data = resampled_img.get_fdata()
            resampled_affine = resampled_img.affine

            return resampled_data, resampled_affine
        except Exception as e:
            print(f"Resampling failed: {e}")
            return None, None  # Indicate failure
        """
        print("work in progrss")

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

    def denoise_file(self, img_data):
        """Placeholder for denoising algorithm."""
        # Implement denoising logic using SimpleITK, SciPy, or other libraries
        print("Denoising placeholder called.")
        return img_data  # Placeholder

    def track_roi(self, img_data):
        """Placeholder for ROI tracking."""
        # Implement ROI tracking logic
        print("ROI Tracking placeholder called.")
        return img_data  # Placeholder

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
        self.playback_speed_label = QLabel(f"Playback Speed: {100} ms")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(10)  # Fastest
        self.speed_slider.setMaximum(500)  # Slowest
        self.speed_slider.setValue(100)
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

            # Create a button with the thumbnail
            thumbnail_button = QPushButton()
            thumbnail_button.setIcon(thumbnail_icon)  # Set as QIcon, not QPixmap
            thumbnail_button.setIconSize(thumbnail_pixmap.size())  # Adjust the size
            thumbnail_button.clicked.connect(lambda: self.select_file_by_thumbnail(file_path))  # Ensure mode switching

            # Store file_path as a property for easier matching
            thumbnail_button.setProperty("file_path", file_path)

            # Create delete button
            delete_button = QPushButton("X")  # Simple 'X' button for delete
            delete_button.setFixedSize(20, 20)  # Set a small size for the delete button
            delete_button.clicked.connect(lambda: self.delete_file(file_path))  # Connect to delete function

            # Create a layout for the thumbnail and delete button
            thumbnail_layout = QHBoxLayout()
            thumbnail_layout.addWidget(thumbnail_button)
            thumbnail_layout.addWidget(delete_button)

            # Create a container widget to hold both buttons
            thumbnail_container = QWidget()
            thumbnail_container.setLayout(thumbnail_layout)

            # Add the thumbnail container directly to the scroll layout
            self.scroll_layout.addWidget(thumbnail_container)

        except Exception as e:
            print(f"Error creating thumbnail for file: {file_path}\n{e}")

    def delete_file(self, file_path):
        """Delete the selected file and remove it from the view without deleting from disk."""
        try:
            filename = os.path.basename(file_path)

            if filename in self.uploaded_files:
                del self.uploaded_files[filename]
            else:
                pass

            # Iterate over the layout to find and remove the thumbnail widget
            for i in reversed(range(self.scroll_layout.count())):
                widget_item = self.scroll_layout.itemAt(i)
                if widget_item is not None:
                    widget = widget_item.widget()  # Get the actual widget from QLayoutItem
                    if widget is not None and widget.layout() is not None:
                        thumbnail_button = widget.layout().itemAt(0).widget()
                        stored_file_path = thumbnail_button.property("file_path")  # Retrieve stored file path

                        # Check if this is the correct widget by comparing the file_path
                        if stored_file_path == file_path:
                            removed_widget = self.scroll_layout.takeAt(i).widget()
                            if removed_widget:
                                removed_widget.setParent(None)  # Remove widget from layout
                                removed_widget.deleteLater()

            # Clear current_file if it matches the deleted file
            if self.current_file == file_path:
                self.current_file = None
                self.img_data = None
                self.img_affine = None
                self.ax.clear()
                self.canvas.draw()
                self.setWindowTitle('TARDIS - No File Selected')

            # Refresh the layout after deleting
            # self.refresh_thumbnail_layout()  # Not needed since we've already removed the widget

        except Exception as e:
            print(f"Error deleting file: {e}")

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
        """Display the preview of modifications."""
        self.preview_manager.show_preview(original, modified)

    # ... Additional methods as necessary ...

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Initial file (optional)
    initial_file = None

    viewer = NiftiViewer(initial_file)
    viewer.setWindowTitle('TARDIS - No File Selected')
    viewer.show()

    sys.exit(app.exec_())
