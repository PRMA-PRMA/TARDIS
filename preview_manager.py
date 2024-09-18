# Preview_manager.py

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import nibabel as nib
import numpy as np
import os

class PreviewManager(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()
        self.original_image = None
        self.modified_image = None

    def setup_ui(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Image display area
        self.image_layout = QHBoxLayout()
        self.original_label = QLabel("Original")
        self.modified_label = QLabel("Modified")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.modified_label.setAlignment(Qt.AlignCenter)
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

        self.setVisible(False)  # Hidden by default

    def show_preview(self, original, modified):
        self.original_image = original
        self.modified_image = modified
        self.display_images()
        self.setVisible(True)
        self.parent.toggle_preview_panel(True)

    def display_images(self):
        """Convert the slices to QPixmap and display them."""
        try:
            # Convert NumPy arrays to QPixmap
            original_pixmap = self.numpy_to_qpixmap(self.original_image)
            modified_pixmap = self.numpy_to_qpixmap(self.modified_image)

            # Check if pixmaps are valid
            if original_pixmap.isNull() or modified_pixmap.isNull():
                QMessageBox.critical(self, "Preview Error", "Failed to generate image previews.")
                return

            # Scale pixmaps to fit the labels while maintaining aspect ratio
            scaled_original = original_pixmap.scaled(
                self.original_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            scaled_modified = modified_pixmap.scaled(
                self.modified_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.original_label.setPixmap(scaled_original)
            self.modified_label.setPixmap(scaled_modified)
        except Exception as e:
            QMessageBox.critical(self, "Display Error", f"Failed to display images:\n{e}")

    def numpy_to_qpixmap(self, img_array):
        """Convert a 2D NumPy array to QPixmap."""
        try:
            # Normalize the image to 0-255
            img_normalized = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
            img_uint8 = (img_normalized * 255).astype(np.uint8)

            # Convert to QImage
            height, width = img_uint8.shape
            bytes_per_line = width
            q_image = QImage(img_uint8.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            return pixmap
        except Exception as e:
            print(f"Error converting NumPy array to QPixmap: {e}")
            return QPixmap()  # Return a null pixmap on failure

    def save_as(self):
        # Save the modified image
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Modified Image", "", "NIfTI Files (*.nii *.nii.gz)")
        if file_path:
            try:
                import nibabel as nib
                # Assuming the parent has the current NIfTI affine
                modified_nifti = nib.Nifti1Image(self.modified_image, self.parent.img_affine)
                nib.save(modified_nifti, file_path)
                QMessageBox.information(self, "Save Successful", f"Modified image saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Failed", f"Failed to save image:\n{e}")

            # Also accept the modification in the main app
            self.parent.accept_modification(self.modified_image)
            self.setVisible(False)

    def reject_modification(self):
        self.setVisible(False)
        self.parent.reject_modification()

    def close_preview(self):
        self.setVisible(False)
        self.parent.toggle_preview_panel(False)
