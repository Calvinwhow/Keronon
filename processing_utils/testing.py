import os
import nibabel as nib
import numpy as np
from nilearn.image import resample_img
import json
import pandas as pd
import csv

class CombineVtas:
    def __init__(self, folder_path=None, mni_brain_path='nifti_utils/bb_bin.nii'):
        self._folder_path = folder_path
        self._mni_brain_path = mni_brain_path

    @property
    def folder_path(self):
        return self._folder_path

    @folder_path.setter
    def folder_path(self, path):
        if os.path.isdir(path):
            self._folder_path = path
        else:
            raise ValueError("Provided path is not a valid directory.")

    @property
    def mni_brain_path(self):
        return self._mni_brain_path

    @mni_brain_path.setter
    def mni_brain_path(self, path):
        if os.path.isfile(path) and path.endswith('.nii'):
            self._mni_brain_path = path
        else:
            raise ValueError("Provided path is not a valid NIfTI file.")

    def combine_niftis(self, output_filename, output_dir):
        if not self._folder_path or not self._mni_brain_path or not output_dir:
            raise ValueError("folder_path, mni_brain_path, and output_dir must be set.")

        mni_brain = nib.load(self._mni_brain_path)
        combined_data = np.zeros(mni_brain.shape)
        single_contact_files = [f for f in os.listdir(self._folder_path) if f.startswith('single_contact') and f.endswith('.nii')]
        
        if not single_contact_files:
            print("No single contact files found. Skipping this patient.")
            return

        for file_name in os.listdir(self._folder_path):
            if file_name.startswith('single_contact') and file_name.endswith('.nii'):
                file_path = os.path.join(self._folder_path, file_name)
                nifti_img = nib.load(file_path)

                resampled_img = resample_img(
                    nifti_img,
                    target_affine=mni_brain.affine,
                    target_shape=mni_brain.shape,
                    interpolation='nearest',
                    force_resample=True,
                    copy_header=True
                )

                combined_data += resampled_img.get_fdata()

        combined_data = np.where(combined_data > 0, 1, 0)

        combined_img = nib.Nifti1Image(combined_data, mni_brain.affine, mni_brain.header)

        output_path = os.path.join(output_dir, output_filename)
        nib.save(combined_img, output_path)

class SpatialCorrelation:
    def __init__(self, target_path=None, vta_path=None, output_folder=None):
        self._target_path = target_path
        self._vta_path = vta_path
        self._output_folder = output_folder

    @property
    def target_path(self):
        return self._target_path

    @target_path.setter
    def target_path(self, path):
        if os.path.isfile(path) and path.endswith('.nii'):
            self._target_path = path
        else:
            raise ValueError("Provided path is not a valid NIfTI file.")

    @property
    def vta_path(self):
        return self._vta_path

    @vta_path.setter
    def vta_path(self, path):
        if os.path.isfile(path) and path.endswith('.nii'):
            self._vta_path = path
        else:
            raise ValueError("Provided path is not a valid NIfTI file.")

    def _update_spatial_correlation_file(self, dot_product, spatial_correl_voxel, spatial_correl_mm3):
        output_path = os.path.join(self._output_folder, 'spatial_correlation.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        file_exists = os.path.isfile(output_path)

        with open(output_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['vta_path', 'dot_product', 'spatial_correlation_voxel', 'spatial_correlation_mm3'])
            writer.writerow([self._vta_path, dot_product, spatial_correl_voxel, spatial_correl_mm3])

    def calculate_correlation(self):
        if not self._target_path or not self._vta_path:
            raise ValueError("target_path and vta_path must be set.")

        target_img = nib.load(self._target_path)
        vta_img = nib.load(self._vta_path)

        target_img = resample_img(
            target_img,
            target_affine=vta_img.affine,
            target_shape=vta_img.shape,
            interpolation='nearest',
            force_resample=True,
            copy_header=True
        )

        target_data = target_img.get_fdata()
        vta_data = vta_img.get_fdata()

        target_data = target_data / np.nanmax(target_data)
        target_data = np.nan_to_num(target_data, nan=0.0)
        if target_data.shape != vta_data.shape:
            raise ValueError("Target and VTA images must have the same shape.")

        dot_product = np.dot(target_data.flatten(), vta_data.flatten())
        voxel_dimensions = vta_img.header.get_zooms()[:3]
        voxel_volume_mm3 = np.prod(voxel_dimensions)
        non_zero_voxels = np.sum(vta_data > 0)
        total_volume_mm3 = non_zero_voxels * voxel_volume_mm3
        spatial_correl_voxel = dot_product / non_zero_voxels
        spatial_correl_mm3 = spatial_correl_voxel / voxel_volume_mm3

        self._update_spatial_correlation_file(dot_product, spatial_correl_voxel, spatial_correl_mm3)

        return dot_product

class VtaAnalysis:
    def __init__(self, folder_path, target_path, dice_path):
        self.combiner = CombineVtas(folder_path=folder_path)
        self.target_path = target_path
        self.dice_path = dice_path

    def run_analysis(self, output_filename):
        self.combiner.combine_niftis(output_filename, self.combiner.folder_path)

        combined_vta_path = os.path.join(self.combiner.folder_path, output_filename)
        correlation_calculator = SpatialCorrelation(target_path=self.target_path, vta_path=combined_vta_path, output_folder=self.dice_path)
        correlation = correlation_calculator.calculate_correlation()

        return correlation