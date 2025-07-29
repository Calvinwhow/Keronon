import os
import json
import numpy as np
from pathlib import Path
from typing import Union, List, Any, Dict

class JsonReader:
    """
    Read a reconstruction JSON file and return the electrode coordinate arrays.
    Can only handle a single electrode per JSON. 
    """
    def __init__(self, json_path: Union[str, Path]):
        self.json_path = Path(json_path).expanduser().resolve()
        self._data: dict[str, Any] | None = None

    ### Private API ###
    def _load(self) -> dict:
        if self._data is None:
            with open(self.json_path, "r") as f:
                self._data = json.load(f)
        return self._data

    def _get(self, *keys) -> Any | None:
        node = self._load()
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return None
            node = node[k]
        return node

    ### Public API ###
    def coords(self) -> List[np.ndarray]:
        """Return a list of (N,3) NumPy arrays—one per electrode."""
        raw = self._get("reco", "mni", "coords_mm")
        if raw is None:
            return []
        return np.asarray(raw)
    
    def elmodels(self) -> List[np.ndarray]:
        """Return a list of (N,3) NumPy arrays—one per electrode."""
        raw = self._get("reco", "props", "elmodel")
        if raw is None:
            return []
        return raw
    
    def segment_lookup(self, electrode_model, segment_lut='contact_segment_labels') -> List[int]:
        '''References the electrode_lut.json file which contains info on which contacts are in which segments.'''
        cwd = os.path.abspath(os.getcwd())
        lut = os.path.join(cwd, 'stim_pyper', 'resources', 'electrode_lut.json')
        with open(lut, 'r') as f:
            lut_dict = json.load(f)
        segments = lut_dict[electrode_model][segment_lut]
        return segments
    
    def pair_contacts_to_segments(self, contact_coords, segments) -> Dict:
        '''Places each contact in a dict with a subdict relating its segment and coordinates.'''
        contact_dict = {}
        for idx in range(0, len(segments)):
            contact_dict[idx] = {segments[idx]: contact_coords[idx]}
        return contact_dict

    def run(self) -> List[np.ndarray]:
        """Orchestration method"""
        coords_arr = self.coords()
        model_str = self.elmodels()

        out = []
        segments = self.segment_lookup(model_str)
        out.append(self.pair_contacts_to_segments(coords_arr, segments))
        return out