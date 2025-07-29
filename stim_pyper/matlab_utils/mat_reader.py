import os
from typing import List, Dict

class MatReader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_reconstructions(self) -> List[Dict]:
        """
        Reads a .mat file and extracts the coordinates based on the file version.
        
        Returns:
            List[Dict]: A list of dictionaries containing directory names and coordinate data.
        """
        json_data = []
        if self.file_path.endswith('.mat'):
            try:
                # Try to use scipy to read the .mat file
                from scipy.io import loadmat
                data = loadmat(self.file_path, simplify_cells=True)
                data = data['reco']['mni']['coords_mm'][:]
                data_dict = {f'coords_{i+1}': data[i] for i in range(len(data))}
                json_data.append({'dir_name': os.path.basename(self.file_path), 'data': data_dict})
            except NotImplementedError:
                # If scipy fails, use h5py for newer .mat file versions
                import h5py
                with h5py.File(self.file_path, 'r') as f:
                    data_refs = f['reco']['mni']['coords_mm'][:]
                    data_dict = {}
                    for i, ref_array in enumerate(data_refs):
                        if isinstance(ref_array[0], h5py.Reference):
                            actual_data = f[ref_array[0]][:].transpose()
                        else:
                            actual_data = ref_array.transpose()
                        data_dict[f'coords_{i+1}'] = actual_data
                    json_data.append({'dir_name': os.path.basename(self.file_path), 'data': data_dict})
            except Exception as e:
                print(f"Error reading {self.file_path}: {e}")
        return json_data
