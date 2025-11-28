"""
Logging utilities for simulation runs.

This module provides functions for logging simulation progress and results.
"""

import logging
from datetime import datetime
from typing import Dict, Any


def setup_logger(name: str = 'spin_decoherence', level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger for simulation runs.
    
    Parameters
    ----------
    name : str
        Logger name
    level : int
        Logging level
        
    Returns
    -------
    logger : logging.Logger
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def log_simulation_start(params: Dict[str, Any], logger: logging.Logger = None) -> None:
    """
    Log simulation start with parameters.
    
    Parameters
    ----------
    params : dict
        Simulation parameters
    logger : logging.Logger, optional
        Logger instance (creates new if None)
    """
    if logger is None:
        logger = setup_logger()
    
    logger.info("=" * 70)
    logger.info("Starting Spin Decoherence Simulation")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"B_rms: {params.get('B_rms', 'N/A')} T")
    logger.info(f"tau_c range: {params.get('tau_c_range', 'N/A')} s")
    logger.info(f"dt: {params.get('dt', 'N/A')} s")
    logger.info(f"T_max: {params.get('T_max', 'N/A')} s")
    logger.info(f"M: {params.get('M', 'N/A')} realizations")
    logger.info("=" * 70)


def log_simulation_end(results: list, logger: logging.Logger = None) -> None:
    """
    Log simulation end with summary statistics.
    
    Parameters
    ----------
    results : list
        Simulation results
    logger : logging.Logger, optional
        Logger instance (creates new if None)
    """
    if logger is None:
        logger = setup_logger()
    
    successful = sum(1 for r in results if r.get('fit_result') is not None)
    
    logger.info("=" * 70)
    logger.info("Simulation Complete")
    logger.info("=" * 70)
    logger.info(f"Total simulations: {len(results)}")
    logger.info(f"Successful fits: {successful}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

