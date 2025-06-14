import numpy as np
from vta_model.model_vta import ModelVta

def process_vta(amplitudes, coordinates, output_path, voxel_size=[0.2, 0.2, 0.2], grid_shape=[71, 71, 71]):
    combined_mask = None

    for amplitude, coord in zip(amplitudes, coordinates):
        if amplitude != 0: 
            vta = ModelVta(center_coord=coord)
            mask = vta.generate_sphere_mask(radius_mm=amplitude)
            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask += np.maximum(combined_mask, mask)

    vta.save_nifti(combined_mask, filename=output_path)
