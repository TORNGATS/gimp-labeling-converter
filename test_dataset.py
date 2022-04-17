

from torch.utils.data import DataLoader
from gimp_labeling_converter import XCFDataset

from torchvision.transforms import ToTensor

dataset = XCFDataset(
    root_dir='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/thermal-segmentor/Pipe_Inspection/pipe_inspection_group01_20201117/label_orig', 
    category={
        'defect' : 1,
        'surface_defect' : 2
    },
    transform=ToTensor(),
    target_transform=ToTensor())

dl = DataLoader(dataset, shuffle=True)

for data in dl:
    print(data)