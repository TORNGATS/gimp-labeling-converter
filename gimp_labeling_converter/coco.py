
import functools
import json
import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Union
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools

from multiprocessing import Process, Manager, Pool, Value

from gimp_labeling_converter.gimp import translator

def generate_categories(anot_dir : str):
    catdir = [ item for item in os.listdir(anot_dir) if os.path.isdir(os.path.join(anot_dir, item)) ]
    cat_list = []
    for i in range(len(catdir)):
        cat_list.append({
            'id' : i+1,
            'name' : catdir[i],
            'supercategory' : 'phm'
        })
    return cat_list

def __coco_core_process(file, images, anotations, helper, image_id, segmentation_id, args):
        filename = os.path.basename(file)
        res = helper(file, **args)
        cs = res.keys()

        image = Image.fromarray(np.uint8(res['original']))
        image_info = pycococreatortools.create_image_info(
            image_id.value, filename, image.size)
        if args['dataset_name'] is not None:
            image_info['dataset_name'] = args['dataset_name']
        images.append(image_info)

        for k in res.keys():
            if k in args['categories']:
                class_id = args['categories'][k]
                category_info = {'id': class_id}
                binary_mask = np.asarray(Image.fromarray(res[k]).convert('1')).astype(np.uint8)
                annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id.value, image_id.value, {**category_info, 'is_crowd' : False}, binary_mask,
                    image.size, tolerance=2)
                if annotation_info is not None:
                    anotations.append(annotation_info)

                segmentation_id.value += 1

        image_id.value += 1

@translator('coco')
def handler_coco__(
    files : Union[str,List[str]], 
    helper : Callable, 
    category : Dict[str, int],
    file_out : str = None, 
    num_workers : int = 1, **kwargs):

    if os.path.isdir(file_out):
        logging.error(f'{file_out} cannot be directory!')
        return

    if category is None or len(category) == 0:
        logging.error('Categories are missing!')
        return

    coco_output = None
    if num_workers <= 1:
        logging.info('Running MS COCO handler in single mode ...')
        coco_output = handler_coco_single__(files, helper, category, **kwargs)
    else:
        logging.info('Running MS COCO handler in parallel mode ...')
        coco_output = handler_coco_parallel__(files, helper, category, num_workers=num_workers, **kwargs)

    if 'annotations' in coco_output.keys():
        for i in range(len(coco_output['annotations'])):
            coco_output['annotations'][i]['image_id'] = i+1

    with open(file_out, 'w') as fout:
        json.dump(coco_output, fout, indent = 4)


def handler_coco_parallel__(
    files : Union[str,List[str]], 
    helper : Callable, 
    category : Dict[str, int],
    dataset_name : str = None,
    description : str = '...',
    url : str = 'https://www.linkedin.com/in/parham-nooralishahi/',
    version : str = '1.0.0',
    year : int = 2022,
    contributor : str = 'phm',
    date_created : str = None,
    licenses : List[Dict] = None,
    num_workers : int = 1, 
    **kwargs : Dict[str,Any]):

    now_str = datetime.utcnow().isoformat(' ')

    _info = {
        "description": description,
        "url": url,
        "version": version,
        "year": year,
        "contributor": contributor,
        "date_created": now_str if date_created is None else date_created
    }

    _licenses = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ] if licenses is None else licenses

    coco_output = {
        "info": _info,
        "licenses": _licenses,
        "categories": category,
        "images": [],
        "annotations": []
    }

    with Manager() as manager:
        fargs = manager.dict()
        images = manager.list()
        anotations = manager.list()
        image_id = manager.Value(int, 1)
        segmentation_id = manager.Value(int, 1)

        fargs.update({
            'categories' : category,
            'dataset_name' : dataset_name,
            'description' : description,
            'url' : url, 
            'version' : version,
            'year' : year,
            'contributor' : contributor,
            'date_created' : date_created,
            **kwargs
        })

        logging.info("Initialize the worker pool ...")
        pool = Pool(processes=num_workers)
        logging.info("Mapping the jobs to the workers ...")
        
        pool.map(functools.partial(__coco_core_process, 
            images=images, 
            anotations=anotations, 
            helper=helper,
            image_id=image_id,
            segmentation_id=segmentation_id,
            args=fargs), files)
        pool.close()
        pool.join()
    
        coco_output['images'].extend(images)
        coco_output['annotations'].extend(anotations)

    return coco_output

def handler_coco_single__(
    files : Union[str,List[str]], 
    helper : Callable, 
    category : Dict[str, int],
    dataset_name : str = None,
    description : str = '...',
    url : str = 'https://www.linkedin.com/in/parham-nooralishahi/',
    version : str = '1.0.0',
    year : int = 2022,
    contributor : str = 'phm',
    date_created : str = None,
    licenses : List[Dict] = None,
    **kwargs : Dict[str,Any]):

    now_str = datetime.utcnow().isoformat(' ')

    _info = {
        "description": description,
        "url": url,
        "version": version,
        "year": year,
        "contributor": contributor,
        "date_created": now_str if date_created is None else date_created
    }

    _licenses = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ] if licenses is None else licenses

    coco_output = {
        "info": _info,
        "licenses": _licenses,
        "categories": category,
        "images": [],
        "annotations": []
    }

    cats = []
    image_id = 1
    segmentation_id = 1

    for file in files:
        filename = os.path.basename(file)
        res = helper(file, **kwargs)
        cs = res.keys()
        cats.extend(cs)

        image = Image.fromarray(np.uint8(res['original']))
        image_info = pycococreatortools.create_image_info(
            image_id, filename, image.size)
        if dataset_name is not None:
            image_info['dataset_name'] = dataset_name
        coco_output['images'].append(image_info)

        for k in res.keys():
            if k in category:
                class_id = category[k]
                category_info = {'id': class_id}
                binary_mask = np.asarray(Image.fromarray(res[k]).convert('1')).astype(np.uint8)
                annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id, image_id, {**category_info, 'is_crowd' : False}, binary_mask,
                    image.size, tolerance=2)
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)
                segmentation_id = segmentation_id + 1

        image_id = image_id + 1

    return coco_output


