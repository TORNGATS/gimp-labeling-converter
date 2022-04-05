
import logging
import phm
from phm.gimp import run_translator

def main():

    config = {
        'dir_in' : '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/thermal-segmentor/Pipe_Inspection/pipe_inspection_group01_20201117/label_orig',
        'dir_out' : '/home/phm/Documents/temp', 
        'handler' : 'mask',
        'binarize' : True,
        'num_workers' : 10
    }

    run_translator(
        config['dir_in'], 
        config['handler'], **config)

if __name__ == "__main__":
    main()