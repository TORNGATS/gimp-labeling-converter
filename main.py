
import logging
import phm
from phm import run_translator

def main():

    # mask
    # config = {
    #     'dir_in' : '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/thermal-segmentor/Pipe_Inspection/pipe_inspection_group01_20201117/label_orig',
    #     'dir_out' : '/home/phm/Documents/temp', 
    #     'handler' : 'mask',
    #     'binarize' : True,
    #     'num_workers' : 10
    # }
    
    # coco
    config = {
        'dir_in' : '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/thermal-segmentor/Pipe_Inspection/pipe_inspection_group01_20201117/label_orig',
        'file_out' : '/home/phm/Documents/temp/phm.coco', 
        'handler' : 'coco',
        'categories' : {
            'defect' : 1, 
            'surface_defect' : 2
        },
        'dataset_name' : 'Pipe_Inspection',
        'description' : 'Test COCO',
        'url' : 'phm.com',
        'version' : '1.0.0',
        'year' : 2022,
        'contributor' : 'Parham Nooralishahi'
    }

    run_translator(
        config['dir_in'], 
        config['handler'], **config)

if __name__ == "__main__":
    main()