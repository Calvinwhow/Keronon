import os 
import h5py
import json
import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Dict

class MatReaderV2:
    '''Class to extract reconstruction data from ea_reconstruction.mat files.'''
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path).expanduser().resolve()
        self._data: Optional[dict] = None
        self._f: Optional[h5py.File] = None
        self._scipy = False

    def _h5_to_dict(self, group):
        '''Converts an h5 file to a dict.'''
        out = {}
        for k, v in group.items():
            out[k] = v if isinstance(v, h5py.Dataset) else self._h5_to_dict(v)
        return out

    def _load(self):
        '''Loads or returns the data (for getting pointers) and h5 file (for referencing w/ pointers)'''
        if self._data is not None:
            return self._data
        if self.file_path.suffix != ".mat":
            raise ValueError("not a .mat file")
        try:
            from scipy.io import loadmat
            self._data = loadmat(self.file_path, simplify_cells=True)
            self._f = None
            self._scipy = True
        except (NotImplementedError, ValueError):
            self._f = h5py.File(self.file_path, "r")
            self._data = self._h5_to_dict(self._f)
        return self._data

    def _get(self, *keys):
        '''Recursion function'''
        node = self._load()
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return None
            node = node[k]
        return node

    def resolve(self, ref):
        '''references the h5py file with retrieved headers to get data'''
        if self._f is None:
            raise RuntimeError("Cannot resolve reference: no open h5py.File")
        return self._f[ref][()].T
    
    def _resolve_scipy_elmodel(self):
        '''Quick function to extract electrode models from old mat files opened with scipy.'''
        model_list = []
        for electrode in self._data['reco']['props']:
            model = electrode['elmodel']
            model_list.append(str(model))
        return model_list
    
    def coords(self) -> List[np.ndarray]:
        d = self._get("reco", "mni", "coords_mm")
        if d is None:
            return []
        
        if self._scipy: # catch if scipy handled old mat file version. return output of dict directly. 
            return d
        
        # Continue with H5Py approach
        arr = d[()] if isinstance(d, h5py.Dataset) else d  # payload

        # single electrode, already numeric
        if arr.dtype != object:
            return [arr]                       # (N,3)

        # multiple electrodes: each row has three HDF5 refs (x,y,z)
        out = []
        for row in arr:                       # shape (n_electrodes, 3)
            xyz = [self.resolve(r) for r in row]   # three (N,) vectors
            out.append(np.column_stack(xyz))       # (N,3)
        return out

    def elmodels(self) -> list[str]:
        '''Recurses through the dict and gets the elmodel string'''
        if self._scipy:
            return self._resolve_scipy_elmodel() 
        
        e = self._get("reco", "props", "elmodel")
        if e is None:
            return []
        
        if self._scipy: # catch if scipy handled old mat file version. return output of dict directly. 
            return e
        
        obj = e[()]
        if obj.dtype == object:                          # multiple refs
            return [''.join(map(chr, self._f[r][()].flatten())) for r in obj.flatten()]
        return [''.join(map(chr, obj.flatten()))]        # single string

    def _get_directional_segments(self, lut_dict, electrode_model):
        '''Get the segments for a directional electrode model'''
        etageidx = lut_dict[electrode_model]['etageidx']
        segments = []
        contact_level = 1
        for idx in etageidx:
            if ':' in idx:
                start, end = map(int, idx.split(':'))
                segments.extend([contact_level] * (end - start + 1))
            else:
                segments.append(contact_level)
            contact_level += 1
        return segments

    def _get_segments(self, lut_dict, electrode_model, segment_lut):
        '''Segment extraction function'''
        try: 
            isdirectional = lut_dict[electrode_model]['isdirected']
            if isdirectional:
                segments = self._get_directional_segments(lut_dict, electrode_model)
            else:
                segments = list(range(1, int(lut_dict[electrode_model][segment_lut]) + 1))
        except KeyError as e:
            raise KeyError(f"Model: {electrode_model} is not implemented in electrode_specs.json")
        except Exception as e:
            raise RuntimeError(f"Error in _get_segments: {e}")
        return segments
    
    def segment_lookup(self, electrode_model, segment_lut='numel') -> List[int]:
        '''References the electrode_lut.json file which contains info on which contacts are in which segments.'''
        cwd = os.path.abspath(os.getcwd())
        lut = os.path.join(cwd, 'stim_pyper', 'resources', 'electrode_specs.json')
        converter = os.path.join(cwd, 'stim_pyper', 'resources', 'elec_converter.json')
        with open(lut, 'r') as f:
            lut_dict = json.load(f)
        with open(converter, 'r') as f:
            converter_dict = json.load(f)
        electrode_model = converter_dict[electrode_model]
        return self._get_segments(lut_dict, electrode_model, segment_lut)
        
    def pair_contacts_to_segments(self, contact_coords, segments) -> Dict:
        '''Places each contact in a dict with a subdict relating its segment and coordinates.'''
        contact_dict = {}
        for idx in range(0, len(segments)):
            contact_dict[idx] = {segments[idx]: contact_coords[idx]}
        return contact_dict
    
    def run(self) -> List[Dict]:
        '''
        Orchestration function. Returns list of dicts (one per electrode).
        First level keys are the integer of the contact.
        The first level values are the subdict with {segment: array[x,y,z]}
        example: [{contact_integer: {{segment_integer: array[x,y,z]} }}]
        '''
        coords_list  = self.coords()
        model_list   = self.elmodels()

        if len(coords_list) != len(model_list):
            raise ValueError("mismatch coords vs models")

        out = []
        for coord, model in zip(coords_list, model_list):
            segments = self.segment_lookup(model)
            out.append(self.pair_contacts_to_segments(coord, segments))
        return out


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
