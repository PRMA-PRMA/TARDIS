# TARDIS: Temporal Analysis and Radiological Display Interactive System

## Overview

TARDIS (Temporal Analysis and Radiological Display Interactive System) is a lightweight, local, web-app-like viewer for NIfTI (Neuroimaging Informatics Technology Initiative) and DICOM files. It supports both CINE (4D) and traditional 3D imaging formats, providing a user-friendly interface for medical imaging professionals and researchers to view and interact with imaging data.

## Features

- **Multi-file Support**: Upload and view multiple NIfTI or DICOM files in a single session.
- **Thumbnail Gallery**: Easy navigation between uploaded files using a scrollable thumbnail sidebar.
- **3D Slice Navigation**: For 3D files, navigate through slices using an interactive slider.
- **CINE Playback**: For 4D imaging data (both NIfTI and DICOM), play through time series data with adjustable playback speed.
- **Dark Mode**: Toggle between dark and light modes for comfortable viewing in different environments.
- **Interactive Controls**: Play, stop, and adjust playback speed for CINE mode.
- **Flexible File Handling**: Supports .nii, .nii.gz, and DICOM file formats.
- **Cross-platform**: Built with PyQt5 for compatibility across different operating systems.

## Installation

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)
- [dcm2niix](https://github.com/rordenlab/dcm2niix) (for DICOM to NIfTI conversion)

### Dependencies

Install the required Python packages using pip:

```
pip install -r requirements.txt
```

To convert DICOM to NIfTI, you will need to install `dcm2niix`:

- For Ubuntu:
  ```
  bash sudo apt-get install dcm2niix
  ```
  
- For macOS (via Homebrew):
  ```
  brew install dcm2niix
  ```
  
- For Windows:
  You can download and install from [here](https://github.com/rordenlab/dcm2niix/releases)

  Alternatively, pip or conda can be used
  ```
  pip install dcm2niix
  ```


### Download

Clone the repository or download the source code:

```
git clone https://github.com/PRMA-PRMA/TARDIS.git
cd TARDIS
```

## Usage

1. Run the application:
   ```
   python tardis.py
   ```

2. To open a NIfTI file on startup, you can modify the `initial_file` variable in the `__main__` section of the script.

3. Use the "Upload New File" button to load additional NIfTI files during runtime.

4. Navigate between loaded files by clicking on their thumbnails in the left sidebar.

5. For 3D files:
   - Use the "Slice Depth" slider to navigate through different slices.

6. For 4D (CINE) files:
   - Use the "Play" and "Stop" buttons to control playback.
   - Adjust the playback speed using the "Playback Speed" slider.

7. Toggle dark mode on/off using the checkbox at the bottom of the controls.

## File Compatibility

- Supports .nii, .nii.gz, and DICOM file formats.
- Handles 2D, 3D, and 4D NIfTI data structures.
- Automatically switches between 3D and CINE modes based on the file structure.

## Troubleshooting

- If you encounter issues loading a file, ensure it's a valid NIfTI format (.nii or .nii.gz) or DICOM format.
- For performance issues with large files, consider downsampling your data before loading.

## Contributing

Contributions to improve TARDIS are welcome. Please feel free to submit pull requests or open issues for bugs and feature requests.

## Acknowledgments

- This project uses [NiBabel](https://nipy.org/nibabel/) for NIfTI file handling.
- UI components are built with [PyQt5](https://www.riverbankcomputing.com/software/pyqt/).
- Plotting functionality is provided by [Matplotlib](https://matplotlib.org/).
- DICOM support and conversion provided by [dcm2niix](https://github.com/rordenlab/dcm2niix)
