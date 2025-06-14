import numpy as np
from vta_model.generate_directional_boundaries import DirectionalVtaBounds

class EvaluateDirectionalVta:
    """
    Class for creating a directional VTA by considering three functions: 
    1) The spherical boundary
    2) Neighbouring planar boundary one
    3) Neighbouring planar boundary two 
    Parameters:
    -----------
    contact_coordinates : array-like
        An list of mni coordinate lists ie) [[x,y,z], [x,y,z], [x,y,z]]. 
        These are used to define VTA geometry
    radius_mm : float, optional
        The radius of the VTA in millimeters. 
        Optional parameter which is not required to intialize for run_optimizer. 
    primary_idx : int, optional
        The index of the list in the contact_coordinates list which the VTA will
        originate from. Defaults to 0.
         
    Notes:
        This class can evaluate how a single VTA eminates from origin coordinates by only calling evalute_spherical_boundary
        This class can evaluate the influence of multiple coordinates on the VTA by calling evaluate_planar_and_spherical_points
    """
    def __init__(self, contact_coordinates, radius_mm=0, primary_idx=0):
        self.radius_mm = radius_mm
        self.boundary_evaluator = DirectionalVtaBounds(contact_coordinates)
        self.contact_coordinates = contact_coordinates
        self.center_coord = self.contact_coordinates[0]
        # Precompute boundary function
        self.boundary_fn = self.boundary_evaluator.composite_boundary_condition(primary_idx)
        
    @property
    def center_coord(self):
        return self._center_coord

    @center_coord.setter
    def center_coord(self, value):
        if not isinstance(value, (list, np.ndarray)):
            raise ValueError("center_coord must be a list or numpy array.")
        if len(value) != 3:
            raise ValueError("center_coord must have exactly 3 values.")
        if not all(isinstance(v, (int, float)) for v in value):
            raise ValueError("All values in center_coord must be numeric.")
        self._center_coord = np.array(value)
          
    @property
    def contact_coordinates(self):
        return self._contact_coordinates

    @contact_coordinates.setter
    def contact_coordinates(self, value):
        if not isinstance(value, list) or not all(isinstance(inner, list) for inner in value):
            raise ValueError("coords must be a list of lists.")
        if len(value) < 2:
            raise ValueError("coords must contain at least two internal lists.")
        if not all(len(inner) == 3 for inner in value):
            raise ValueError("Each internal list in coords must have exactly 3 values.")
        if not all(all(isinstance(v, (int, float)) for v in inner) for inner in value):
            raise ValueError("All values in the internal lists of coords must be numeric.")
        self._contact_coordinates = np.array(value)

    @property
    def radius_mm(self):
        return self._radius_mm

    @radius_mm.setter
    def radius_mm(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("radius_mm must be a numeric value.")
        if value <= 0:
            raise ValueError("radius_mm must be a positive value.")
        self._radius_mm = value
        
    def evalute_spherical_boundary(self, points, center_coord, radius_mm):
        '''Quick function to calculate points within a sphere'''
        distances = np.linalg.norm(points - center_coord, axis=1)
        mask_flat = (distances <= radius_mm)
        return mask_flat
            
    def evaluate_planar_boundary(self, points):
        '''Quick function to calculate points within composite planes'''
        mask_flat = self.boundary_fn(points)
        return mask_flat
    
    def evaluate_planar_and_spherical_points(self, points):
        '''
        Check points explicitly against both the radius and directional boundary conditions.
        Returns boolean array indicating if points pass both conditions.
        '''
        points = np.array(points)
        within_radius = self.evalute_spherical_boundary(points, self.center_coord, self.radius_mm)
        within_planes = self.evaluate_planar_boundary(points)
        mask_flat =  (within_radius & within_planes).astype(np.int16)
        return mask_flat
