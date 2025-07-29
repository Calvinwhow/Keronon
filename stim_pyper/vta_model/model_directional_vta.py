from stim_pyper.vta_model.model_vta import ModelVta
from stim_pyper.vta_model.evaluate_directional_vta import EvaluateDirectionalVta

class ModelDirectionalVta(ModelVta):
    """
    A model for simulating Volume of Tissue Activated (VTA) with support for both undirected and directed VTAs.
    This class extends the `ModelVta` base class to handle VTAs based on the input coordinates. If the input 
    coordinates consist of a single point (e.g., [[x, y, z]]), the model generates an undirected VTA. If the 
    input coordinates include multiple points (e.g., [[x1, y1, z1], [x2, y2, z2]]), the model generates a 
    directional VTA based on the provided directional boundaries.
    Attributes:
        contact_coordinates (list of lists): A list of coordinate points defining the VTA. For undirected VTAs, this should 
            be a single point in the format [[x, y, z]]. For directional VTAs, this should include multiple 
            points in the format [[x1, y1, z1], [x2, y2, z2], ...].
        origin_index (int): The index of the coordinate to use as the center of the VTA. Defaults to 0.
        voxel_size (list of float): The size of each voxel in the grid, specified as [x, y, z]. Defaults to [0.2, 0.2, 0.2].
        grid_shape (list of int): The shape of the grid, specified as [x, y, z]. Defaults to [71, 71, 71].
        output_path (str): The directory where the generated VTA file will be saved. Defaults to the current directory.
        fname (str): The name of the output file for the generated VTA. Defaults to "directional_vta.nii.gz".
    Methods:
        generate_directional_mask(radius_mm):
            Generates a mask that satisfies both the spherical radius and directional boundary conditions.
        choose_mask(radius_mm):
            Chooses the appropriate mask (spherical or directional) based on the number of input coordinates.
        run(radius_mm):
            Generates the VTA mask and saves it as a NIfTI file.
    Note:
        - All input coordinates should be provided as a list of lists. For example:
            - Undirected VTA: [[x, y, z]]
            - Directed VTA: [[x1, y1, z1], [x2, y2, z2], ...]
    """
    
    def __init__(self, contact_coordinates, origin_index=0,
                 voxel_size=[0.2, 0.2, 0.2], grid_shape=[71, 71, 71], 
                 output_path=".", fname="directional_vta.nii.gz"):
        self.contact_coordinates = contact_coordinates
        self.origin_index = origin_index  
        super().__init__(center_coord=contact_coordinates[origin_index], 
                         voxel_size=voxel_size, grid_shape=grid_shape, 
                         output_path=output_path, fname=fname)
    
    @property
    def contact_coordinates(self):
        return self._contact_coordinates

    @contact_coordinates.setter
    def contact_coordinates(self, value):
        if not isinstance(value, list) or not all(isinstance(coord, list) and len(coord) == 3 for coord in value):
            raise ValueError("contact_coordinates must be a list of lists, where each inner list contains exactly three numerical values.")
        self._contact_coordinates = value
    
    def generate_directional_mask(self, radius_mm):
        '''Generate mask satisfying both radius and directional boundary conditions.'''
        # Get mask from spherical boundary
        sphere_mask = super().generate_sphere_mask(radius_mm).flatten()
        evaluator = EvaluateDirectionalVta(contact_coordinates=self.contact_coordinates, radius_mm=radius_mm, primary_idx=self.origin_index)
        flat_mask = evaluator.evaluate_planar_and_spherical_points(self.coordinates)
        return flat_mask.reshape(self.grid_shape)
    
    def choose_mask(self, radius_mm):
        if len(self.contact_coordinates) == 1:
            mask = self.generate_sphere_mask(radius_mm)
        else:
            mask = self.generate_directional_mask(radius_mm)
        return mask
            
    def run(self, radius_mm):
        """
        Generates the Volume of Tissue Activated (VTA) and saves it as a NIfTI file.
        Runs either directed or undirected, depending on number of coordinates in array.
       
        Params:
            radius_mm (float): The radius in millimeters used to generate the VTA mask.
        """
        mask = self.choose_mask(radius_mm)
        self.save_nifti(mask)
