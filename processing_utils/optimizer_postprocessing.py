import os
import json
from vta_model.model_directional_vta import ModelDirectionalVta

def process_vta(amplitudes, coordinates, side, output_dir, voxel_size=[0.2, 0.2, 0.2], grid_shape=[71, 71, 71]):
    filenames = []
    stimparams = os.path.join(output_dir, f'stimparams_elec_{side}.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(stimparams, 'w') as json_file:
        json.dump(amplitudes.tolist(), json_file)

    for amplitude, coord in zip(amplitudes, coordinates):
        if amplitude != 0: 
            coord_tuple = tuple(coord)
            index = [tuple(c) for c in coordinates].index(coord_tuple)
            updated_fname = f'single_contact_elec_{side}_contact_{index}.nii'            
            coord = coord.tolist()

            if index in [1, 2, 3]:
                group_indices = [1, 2, 3]
            elif index in [4, 5, 6]:
                group_indices = [4, 5, 6]
            else:
                group_indices = []
            if group_indices and len(coordinates) > 4:
                adjacent_coords = [coordinates[i].tolist() for i in group_indices if i != index]
                contact_coordinates = [coord] + adjacent_coords
            else:
                contact_coordinates = [coord]
            vta = ModelDirectionalVta(contact_coordinates=contact_coordinates,
                voxel_size=voxel_size,
                grid_shape=grid_shape,
                output_path=output_dir,
                fname=updated_fname)
            vta.run(radius_mm=amplitude)
            filenames.append(vta.fpath)

