
import os
import logging
import functools
import numpy as np
import multiprocessing
from PIL import Image
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

from gimp_labeling_converter.gimp import translator 

def __save_gimp_layers(file: str, helper, file_out : str = None, **kwargs):
    """
    Save the layers of xcf file in seperated folders

    Args:
        file (str): the path of xcf files
        args (dict): program arguments
    """

    dout = file_out if file_out else os.path.pardir(file)
    res = helper(file, **kwargs)
    filename = os.path.basename(file)
    fname = filename.split(".")[0]
    for key in res.keys():
        odir = os.path.join(dout, key)
        Path(odir).mkdir(parents=True, exist_ok=True)
        out = res[key]
        out_img = Image.fromarray(np.uint8(out))
        out_img.save(os.path.join(odir, fname) + ".png")

@translator('mask')
def handler_mask__(
    files : Union[str,List[str]], 
    helper : Callable, 
    num_workers : int = 1, 
    **kwargs : Dict[str,Any]):
    
    logging.info("Initialize the worker pool ...")
    pool = multiprocessing.Pool(processes=num_workers)
    logging.info("Mapping the jobs to the workers ...")

    inputs = [files,] if type(files) is str else files
    pool.map(functools.partial(__save_gimp_layers, helper=helper, **kwargs), inputs)
    pool.close()
    pool.join()
