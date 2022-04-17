
import os
import glob2
import logging
import functools
import numpy as np
import multiprocessing

from PIL import Image
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from gimpformats.gimpXcfDocument import GimpDocument

class phmError(Exception):
    def __init__(self, message : str, *args: object) -> None:
        super().__init__(*args)
        self.message = message

def exception_logger(func):
    """
    A decorator that wraps the passed in function and logs exceptions should one occur.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except phmError as ex:
            # log the exception
            logging.exception(f'{func.__name__} | {ex.message}')
            raise ex
    return wrapper

translator_handlers = {}
def translator(name):
    def __embed_func(func):
        global translator_handlers
        translator_handlers[name] = func

    return __embed_func

def list_handlers() -> List[str]:
    return translator_handlers.keys()

__file_formats__ = ['jpg', 'jpeg', 'png', 'tiff', 'bmp']

@exception_logger
def generate_cmap(
    file : str,
    helper : Callable,
    category : Dict[str, int], 
    **kwargs) -> Dict[str, Any]:

    res = {}
    layers = helper(file, category, **kwargs)
    res['original'] = layers.pop('original')

    layers = list(layers.values())
    layers = np.dstack(layers)
    res['target'] = np.amax(layers, axis=2)
    return res

@exception_logger
def gimp_helper(
    file : str,
    category : Dict[str, int] = None,
    **kwargs) -> Dict[str, Any]:

    # Argument initialization and checking
    if not Path(file).is_file():
        raise phmError(message=f'The file ({file}) does not exist!')
    if not file.endswith(".xcf"):
        raise phmError(message=f'The file format ({file}) is not supported!')
    #####################
    
    # Load the GIMP file
    gimp = GimpDocument(file)
    layers = gimp.layers
    num_layers = len(layers)

    logging.info(f"Processing (Layers [{num_layers}]) {file}")
    if len(layers) < 2:
        logging.error(f'The xcf file does not have sufficient class layers.')
        return
    
    filename = os.path.basename(file)
    result = {}
    logging.info(f'Processing {filename} ...')
    # Go through the layers
    for layer in layers:
        if not layer.isGroup:
            fex = layer.name.split('.')[-1].lower()
            # Check if the layer is a group layers
            if fex in __file_formats__:
                result['original'] = layer.image
            else:
                img = layer.image
                # Select only layers with 4 channels.
                if img.mode in ("RGBA", "LA") or (
                    img.mode == "P" and "transparency" in img.info):
                    channels = img.split()
                    img_ch = channels[-1].convert('1')
                    pix_value = 255
                    if category is not None and len(category) != 0:
                        pix_value = category[layer.name]
                    result[layer.name] = np.where(np.asarray(img_ch) != 0, pix_value, 0).astype(np.int8)
    return result

@exception_logger
def run_translator(dir_path : str, trans_name : str = 'mask', **kwargs):

    if not trans_name in translator_handlers:
        raise phmError(message=f'{trans_name} is not defined!')

    def __list_files(dir : str):
        if not os.path.isdir(dir):
            raise phmError(message=f'{dir} is not a valid directory!')
        files = list(filter(lambda x: not os.path.isdir(x), glob2.glob(os.path.join(dir, "*.xcf"))))
        return files
    
    files = __list_files(dir_path)
    logging.info(f'{trans_name} handler is started ...')
    translator_handlers[trans_name](files, gimp_helper, **kwargs)
    logging.info(f'{trans_name} handler is completed!')
