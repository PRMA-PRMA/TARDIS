import os
import logging
import subprocess
import shutil

def handle_file_upload(file_path):
    def is_dicom_file(filepath):
        """Check if a file is a DICOM file by reading the magic number."""
        try:
            with open(filepath, 'rb') as file:
                file.seek(128)
                magic_number = file.read(4)
            return magic_number == b'DICM'
        except Exception as e:
            logging.error(f"Error reading file {filepath}: {str(e)}")
            return False

    def convert_dicom_to_nifti(dicom_file_path, output_directory, compression=True, reorient=True, verbose=False):
        """Convert the provided DICOM file to NIfTI format."""
        try:
            # Create the output directory if it doesn't exist
            os.makedirs(output_directory, exist_ok=True)

            # Run dcm2niix to convert the DICOM file to NIfTI
            run_dcm2niix(dicom_file_path, output_directory, compression=compression, reorient=reorient, verbose=verbose)

            # Return the path to the first NIfTI file in the output directory
            return get_first_nifti_file(output_directory)

        except Exception as e:
            logging.error(f"Failed to convert DICOM to NIfTI: {str(e)}")
            return None

    def run_dcm2niix(input_file, output_dir, compression=True, reorient=True, verbose=False):
        """Run dcm2niix to convert DICOM to NIfTI format."""
        compression_flag = '-z y' if compression else '-z n'
        reorient_flag = '' if reorient else '-i n'
        verbose_flag = '-v 1' if verbose else '-v 0'

        command = f'dcm2niix {compression_flag} {reorient_flag} {verbose_flag} -o "{output_dir}" "{input_file}"'
        try:
            subprocess.run(command, check=True, shell=True)
            logging.info(f"Successfully converted {input_file} to NIfTI format.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to convert {input_file}. Error: {str(e)}")
            raise

    def get_first_nifti_file(directory):
        """Helper function to get the first NIfTI file in the directory."""
        for file_name in os.listdir(directory):
            if file_name.endswith(('.nii', '.nii.gz')):
                return os.path.join(directory, file_name)
        raise FileNotFoundError("No NIfTI file found in the output directory.")

    """Determine whether the file is a NIfTI or DICOM file and handle accordingly."""
    if is_dicom_file(file_path):
        # Convert DICOM to NIfTI and return the path to the NIfTI file
        output_directory = "path_to_save_converted_nifti_files"
        return convert_dicom_to_nifti(file_path, output_directory)
    else:
        # Return the NIfTI file path as is
        return file_path

def clean_nifti_dir(dir_path = "path_to_save_converted_nifti_files"):
    shutil.rmtree(dir_path)

def extract_slice(nii_file):
    """Extract a single slice from the NIfTI file."""
    data = nii_file.get_fdata()
    shape = data.shape
    if len(shape) == 2:  # 2D image
        return data
    elif len(shape) == 3:  # 3D image
        return data[:, :, shape[2] // 2]
    elif len(shape) == 4 and shape[3] == 1:  # 3D image disguised as 4D
        return data[:, :, shape[2] // 2, 0]
    elif len(shape) == 4:  # 4D image (CINE)
        return data[:, :, shape[2] // 2, 0]
    else:
        raise ValueError(f"Invalid shape {shape} for image data")
