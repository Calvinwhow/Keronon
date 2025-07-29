import os 
import numpy as np
import nibabel as nib

class ModelVta:
    def __init__(self, center_coord, voxel_size=[0.2, 0.2, 0.2], grid_shape=[71, 71, 71], output_path=".",  fname="optimized_vta.nii.gz"):
        """
        Initialize the ModelVta object.

        Parameters
        ----------
        center_coord : list or array-like
            The center coordinates of the grid in 3D space (x, y, z).
        voxel_size : list or array-like, optional
            The size of each voxel in the grid in mm (default is [0.2, 0.2, 0.2]).
        grid_shape : list or array-like, optional
            The shape of the grid (number of voxels in each dimension, default is [71, 71, 71]).
        output_path : str, optional
            The path to save the output NIFTI file (default is current directory).
        fname : str, optional
            The filename of the output NIFTI file. 

        Raises
        ------
        ValueError
            If any dimension of the grid shape is not an odd number.
        """
        self.center_coord = center_coord
        self.voxel_size = voxel_size
        self.grid_shape = grid_shape
        self.output_path = output_path
        self.fpath = fname

    @property
    def center_coord(self):
        return self._center_coord

    @center_coord.setter
    def center_coord(self, value):
        if not (isinstance(value, list) and len(value) == 3):
            raise ValueError("coords must be a list of 3 values")
        self._center_coord = np.array(value, dtype=float)

    @property
    def voxel_size(self):
        return self._voxel_size

    @voxel_size.setter
    def voxel_size(self, value):
        value = np.array(value, dtype=float)
        if np.any(value <= 0):
            raise ValueError("Voxel sizes must be positive.")
        self._voxel_size = value

    @property
    def grid_shape(self):
        return self._grid_shape

    @grid_shape.setter
    def grid_shape(self, value):
        value = np.array(value, dtype=int)
        if np.any(value % 2 == 0):
            raise ValueError("Grid dimensions must be odd.")
        self._grid_shape = value

    @property
    def output_path(self):
        return self._output_path

    @output_path.setter
    def output_path(self, path):
        path = os.path.abspath(path)
        os.makedirs(path, exist_ok=True)
        self._output_path = path
    
    @property
    def fpath(self):
        return self._fpath
    
    @fpath.setter
    def fpath(self, fname):
        self._fpath = os.path.join(self.output_path, fname)

    @property
    def affine(self):
        offset = self.center_coord - self.voxel_size * ((self.grid_shape - 1) / 2)
        return np.array([
            [self.voxel_size[0], 0, 0, offset[0]],
            [0, self.voxel_size[1], 0, offset[1]],
            [0, 0, self.voxel_size[2], offset[2]],
            [0, 0, 0, 1]
        ])

    @property
    def coordinates(self):
        half_extent = (self.grid_shape - 1) / 2
        offsets = [np.arange(-h, h+1) for h in half_extent]
        grid_offsets = np.meshgrid(*offsets, indexing='ij')
        stacked_offsets = np.stack(grid_offsets, axis=-1)
        scaled_offsets = stacked_offsets * self.voxel_size
        return scaled_offsets.reshape(-1, 3) + self.center_coord
    
    def modify_header(self, header):
        '''Assign headers to nifti'''   
        header.set_zooms(self.voxel_size)
        header.set_sform(self.affine, code=1)
        header.set_qform(self.affine, code=1)
        header.set_data_dtype(np.int16)
        header['descrip'] = 'Binary VTA mask (int16)'
        return header

    def generate_sphere_mask(self, radius_mm):
        '''Generate a spherical e-field based on e-field radius'''
        distances = np.linalg.norm(self.coordinates - self.center_coord, axis=1)
        mask_flat = (distances <= radius_mm).astype(np.int16)
        return mask_flat.reshape(self.grid_shape)

    def save_nifti(self, data):
        '''Will save a generated nifti'''
        nii_img = nib.Nifti1Image(data, self.affine)
        self.modify_header(nii_img.header)
        nib.save(nii_img, self.fpath)

    def run(self, radius_mm):
        '''Will generate a nifti and save it'''
        mask = self.generate_sphere_mask(radius_mm)
        self.save_nifti(mask)