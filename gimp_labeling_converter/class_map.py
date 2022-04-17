
import os
import logging
import functools
from pathlib import Path
import numpy as np
import multiprocessing

from PIL import Image
from typing import Any, Callable, Dict, List, Union
from gimp_labeling_converter.gimp import generate_cmap, translator 

def __save_cmask_target(
    file: str, 
    helper : Callable, 
    file_out : str = None, 
    category : Dict[str, int] = None, 
    **kwargs):

    dout = file_out if file_out else os.path.pardir(file)
    
    orig_dir = os.path.join(dout, 'original')
    Path(orig_dir).mkdir(parents=True, exist_ok=True)
    target_dir = os.path.join(dout, 'target')
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    res = helper(file, category, **kwargs)
    filename = os.path.basename(file)
    fname = filename.split(".")[0]

    res = generate_cmap(file, helper, category, **kwargs)
    orig = res['original']
    out = res['target']

    orig.save(os.path.join(orig_dir, fname) + ".png")
    
    out_img = Image.fromarray(np.uint8(out))
    out_img.save(os.path.join(target_dir, fname) + ".png")
    
    # orig = res.pop('original')

    # # Concatenate the class layers
    # layers = list(res.values())
    # layers = np.dstack(layers)
    # out = np.amax(layers, axis=2)

@translator('class_map')
def handler_class_map__(
    files : Union[str,List[str]], 
    helper : Callable, 
    category : Dict[str, int],
    num_workers : int = 1, 
    **kwargs : Dict[str,Any]):
    
    if category is None or len(category) == 0:
        logging.error('Categories are missing!')
        return

    logging.info("Initialize the worker pool ...")
    pool = multiprocessing.Pool(processes=num_workers)
    logging.info("Mapping the jobs to the workers ...")

    inputs = [files,] if type(files) is str else files
    pool.map(functools.partial(__save_cmask_target, helper=helper, category=category, **kwargs), inputs)
    pool.close()
    pool.join()