import sys
import nibabel as nib
import numpy as np
import SimpleITK as sitk  # Ensure SimpleITK is installed for this
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QMainWindow, QApplication, QVBoxLayout, QPushButton, QSlider, QWidget, QLabel,
                             QHBoxLayout, QCheckBox, QScrollArea, QGridLayout, QFileDialog, QSizePolicy,
                             QDesktopWidget, QMenuBar, QAction, QMessageBox, QDialog, QLineEdit)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QPixmap
import os
from app_utils import handle_file_upload, extract_slice, clean_nifti_dir


class ResamplingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resampling Options")
        self.setModal(True)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Resampling Factor Input
        factor_layout = QHBoxLayout()
        factor_label = QLabel("Resampling Factor:")
        self.factor_input = QLineEdit()
        self.factor_input.setPlaceholderText("e.g., 2")
        factor_layout.addWidget(factor_label)
        factor_layout.addWidget(self.factor_input)
        layout.addLayout(factor_layout)

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


# Similarly, create RegistrationDialog, DenoisingDialog, and ROTrackingDialog


class NiftiViewer(QMainWindow):
    def __init__(self, initial_file=None):
        super().__init__()

        # Dictionary to store all uploaded files (filename -> NIfTI object)
        self.current_file = None  # Initialize current file
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
        self.scroll_area.setFixedWidth(250)  # Set a fixed width for the thumbnail area
        self.main_layout.addWidget(self.scroll_area)
        self.thumbnail_widgets = {}  # Dictionary to store the thumbnails with their corresponding filenames

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
        # Implement RegistrationDialog similar to ResamplingDialog
        QMessageBox.information(self, "Info", "Registration dialog not implemented yet.")

    def open_denoising_dialog(self):
        # Implement DenoisingDialog similar to ResamplingDialog
        QMessageBox.information(self, "Info", "Denoising dialog not implemented yet.")

    def open_roi_tracking_dialog(self):
        # Implement ROTrackingDialog similar to ResamplingDialog
        QMessageBox.information(self, "Info", "ROI Tracking dialog not implemented yet.")

    def apply_resampling(self, factor):
        if not self.current_file:
            QMessageBox.warning(self, "No File Loaded", "Please load a file to apply modifications.")
            return

        try:
            original_data = self.img_data.copy()
            modified_data = self.resample_algorithm(original_data, factor)
            self.img_data = modified_data
            self.update_image()
            QMessageBox.information(self, "Resampling", "Resampling applied successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply resampling:\n{e}")

    def apply_intensity_normalization(self, min_val, max_val):
        if not self.current_file:
            QMessageBox.warning(self, "No File Loaded", "Please load a file to apply modifications.")
            return

        try:
            original_data = self.img_data.copy()
            modified_data = self.normalize_intensity(original_data, min_val, max_val)
            self.img_data = modified_data
            self.update_image()
            QMessageBox.information(self, "Intensity Normalization", "Intensity normalization applied successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply intensity normalization:\n{e}")

    def apply_registration(self, reference_file):
        if not self.current_file:
            QMessageBox.warning(self, "No File Loaded", "Please load a file to apply modifications.")
            return

        try:
            reference_nii = nib.load(reference_file)
            reference_data = reference_nii.get_fdata()
            original_data = self.img_data.copy()
            modified_data = self.register_file(original_data, reference_data)
            self.img_data = modified_data
            self.update_image()
            QMessageBox.information(self, "Registration", "Registration applied successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply registration:\n{e}")

    def apply_denoising(self, denoising_params):
        if not self.current_file:
            QMessageBox.warning(self, "No File Loaded", "Please load a file to apply modifications.")
            return

        try:
            original_data = self.img_data.copy()
            modified_data = self.denoise_file(original_data)
            self.img_data = modified_data
            self.update_image()
            QMessageBox.information(self, "Denoising", "Denoising applied successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply denoising:\n{e}")

    def apply_roi_tracking(self, roi_params):
        if not self.current_file:
            QMessageBox.warning(self, "No File Loaded", "Please load a file to apply modifications.")
            return

        try:
            original_data = self.img_data.copy()
            modified_data = self.track_roi(original_data)
            self.img_data = modified_data
            self.update_image()
            QMessageBox.information(self, "ROI Tracking", "ROI tracking applied successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply ROI tracking:\n{e}")

    def resample_algorithm(self, img_data, factor):
        """Placeholder for resampling algorithm."""
        # Example implementation using SimpleITK
        try:
            img = sitk.GetImageFromArray(img_data)
            original_spacing = img.GetSpacing()
            new_spacing = tuple([s / factor for s in original_spacing])
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(new_spacing)
            resampler.SetSize([int(sz * f) for sz, f in zip(img.GetSize(), [factor]*len(img.GetSize()))])
            resampler.SetInterpolator(sitk.sitkLinear)
            resampled_img = resampler.Execute(img)
            resampled_data = sitk.GetArrayFromImage(resampled_img)
            return resampled_data
        except Exception as e:
            print(f"Resampling failed: {e}")
            return img_data  # Return original data on failure

    def normalize_intensity(self, img_data, min_val, max_val):
        """Placeholder for intensity normalization."""
        try:
            img_min = np.min(img_data)
            img_max = np.max(img_data)
            normalized = (img_data - img_min) / (img_max - img_min)  # Scale to [0,1]
            normalized = normalized * (max_val - min_val) + min_val  # Scale to [min_val, max_val]
            return normalized
        except Exception as e:
            print(f"Intensity normalization failed: {e}")
            return img_data  # Return original data on failure

    def register_file(self, img_data, reference_data):
        """Placeholder for image registration."""
        # Implement registration logic using SimpleITK or other libraries
        print("Registration placeholder called.")
        return img_data  # Placeholder

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
            self.add_thumbnail(file_path)  # Now, let add_thumbnail handle the addition to uploaded_files

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
        """Create a thumbnail for the file and add it to the sidebar with a delete button."""
        try:
            filename = os.path.basename(file_path)

            # Only add the file if it hasn't already been added
            if filename in self.uploaded_files:
                print("uploaded Files:", self.uploaded_files)
                #print(f"File {filename} was previously added. Ensuring full cleanup before re-adding.")
                self.delete_file(file_path)  # Clean up if the file was previously added

            # Now, add the file to the dictionary of uploaded files after the check
            nii_file = nib.load(file_path)
            self.uploaded_files[filename] = nii_file
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
            row = len(self.uploaded_files) - 1
            self.scroll_layout.addWidget(thumbnail_container, row, 0)

        except Exception as e:
            print(f"Error creating thumbnail for file: {file_path}\n{e}")

    def delete_file(self, file_path):
        """Delete the selected file and remove it from the view without deleting from disk."""
        try:
            #print(f"Attempting to delete file: {file_path}")
            # Get the filename and remove the file from the dictionary of uploaded files
            filename = os.path.basename(file_path)

            if filename in self.uploaded_files:
                del self.uploaded_files[filename]
                #print(f"Removed {filename} from uploaded_files")
            else:
                pass
                #print(f"File {filename} not found in uploaded_files")

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
                            #print(f"Match found! Removing widget for {file_path}")
                            removed_widget = self.scroll_layout.takeAt(i).widget()
                            if removed_widget:
                                removed_widget.setParent(None)  # Remove widget from layout
                                removed_widget.deleteLater()  # Schedule it for deletion
                                #print(f"Widget removed for {file_path}")

            # Clear current_file if it matches the deleted file
            if self.current_file == file_path:
                self.current_file = None
                self.ax.clear()
                self.canvas.draw()
                self.setWindowTitle('NIfTI Viewer - No File Selected')
                print(f"Cleared display for {file_path}")

            # Refresh the layout after deleting
            self.refresh_thumbnail_layout()

        except Exception as e:
            print(f"Error deleting file: {e}")

    def refresh_thumbnail_layout(self):
        """Clear the layout and rebuild it with the remaining thumbnails."""
        # Clear all the items from the layout
        for i in reversed(range(self.scroll_layout.count())):
            item = self.scroll_layout.takeAt(i)
            if item.widget():
                item.widget().deleteLater()

        # Rebuild the layout by re-adding all the current thumbnails
        for row, filename in enumerate(self.uploaded_files.keys()):
            self.add_thumbnail(self.uploaded_files[filename].get_filename())  # Re-add the thumbnail

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
