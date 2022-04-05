
import json
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Union
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools

from phm.gimp import translator

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

# CATEGORIES = [
#     {
#         'id': 1,
#         'name': 'square',
#         'supercategory': 'shape',
#     },
#     {
#         'id': 2,
#         'name': 'circle',
#         'supercategory': 'shape',
#     },
#     {
#         'id': 3,
#         'name': 'triangle',
#         'supercategory': 'shape',
#     },
# ]

# CATEGORIES = {
#     'sqaure' : 1,
#     'circle' : 2,
#     'triangle' : 3
# }

@translator('coco')
def handler_coco__(
    files : Union[str,List[str]], 
    helper : Callable, 
    categories : List[str],
    file_out : str = None,
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
        "categories": categories,
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
            if k in categories:
                class_id = categories[k]
                category_info = {'id': class_id}
                binary_mask = np.asarray(Image.fromarray(res[k]).convert('1')).astype(np.uint8)
                annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id, image_id, {**category_info, 'is_crowd' : False}, binary_mask,
                    image.size, tolerance=2)
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)
                segmentation_id = segmentation_id + 1

        image_id = image_id + 1

    with open(file_out, 'w') as fout:
        json.dump(coco_output, fout, indent = 4)

