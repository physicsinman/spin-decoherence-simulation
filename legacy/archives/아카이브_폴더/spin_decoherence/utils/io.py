"""
File I/O utilities for simulation results.

This module provides functions for saving and loading simulation results.
"""

import json
import os
from datetime import datetime
from typing import Any, Union, List, Dict


def make_json_serializable(obj: Any) -> Any:
    """
    Convert object to JSON-serializable format.
    
    Parameters
    ----------
    obj : Any
        Object to convert
        
    Returns
    -------
    serializable : Any
        JSON-serializable representation
    """
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.complexfloating):
        return {'real': float(obj.real), 'imag': float(obj.imag)}
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj


def save_results(results: Union[List[Dict], Dict], output_dir: str = None, filename: str = None) -> str:
    """
    Save simulation results to JSON file.
    
    Parameters
    ----------
    results : list or dict
        Simulation results
    output_dir : str, optional
        Output directory (default: 'results')
    filename : str, optional
        Output filename (default: auto-generated with timestamp)
        
    Returns
    -------
    filepath : str
        Path to saved file
    """
    if output_dir is None:
        output_dir = 'results'
    
    os.makedirs(output_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_results_{timestamp}.json"
    
    filepath = os.path.join(output_dir, filename)
    
    # Convert to JSON-serializable format
    json_results = make_json_serializable(results)
    
    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def load_results(result_file: str) -> List[Dict]:
    """
    Load simulation results from JSON file.
    
    Parameters
    ----------
    result_file : str
        Path to JSON results file
        
    Returns
    -------
    results : list of dict
        List of result dictionaries
    """
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # Handle both single result and list of results
    if isinstance(results, dict):
        results = [results]
    
    return results

