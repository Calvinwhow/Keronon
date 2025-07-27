import numpy as np

class DirectionalVtaBounds:
    def __init__(self, contact_coordinates):
        '''Receives coordinates of contacts'''
        self.contact_coordinates = np.array(contact_coordinates)

    def _get_midpoint(self, origin, reference):
        '''Get geometric midpoint of two coordinates'''
        return (origin + reference) / 2
    
    def _bisector_plane(self, origin, reference):
        '''
        Explicit algebraic coefficients of bisector plane equation Ax+By+Cz+D=0,
        where A,B,C are the difference in each dimension of Reference - Origin
        Then, D is D = -(Ax_m + By_m + Cz_m),
        where x_m, y_m, and z_m are coordinates of the midpoint.
        A full derivation of this is left for the user :).
        '''
        midpoint = self._get_midpoint(origin, reference)
        normal = reference - origin
        A, B, C = normal
        D = -np.dot(normal, midpoint)
        return A, B, C, D  # Coefficients explicitly

    def evaluate_plane(self, plane_coeffs, points):
        '''
        Evaluates the plane equation (Ax+By+Cz+D) for an array of points
        By convention, negative values are closer to P0, while positive are closer to P1, and 0 is on the plane.
        '''
        A, B, C, D = plane_coeffs
        return np.dot(points, np.array([A, B, C])) + D
    
    def composite_boundary_condition(self, origin_idx=0):
        '''
        Composite condition precomputing both bisector planes
        If either value is positive, it is beyond the plane. 
        '''
        primary = self.contact_coordinates[origin_idx]
        neighbors = [self.contact_coordinates[i] for i in range(3) if i != origin_idx]
        planes = [self._bisector_plane(primary, neighbor) for neighbor in neighbors]

        def boundary_fn(points):
            '''Vectorized single-check boundary condition'''
            results = np.maximum.reduce([self.evaluate_plane(plane, points) for plane in planes])
            return results < 0
        return boundary_fn
