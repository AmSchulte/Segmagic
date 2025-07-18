from dataset_ml import TrainImage, ImageDataset
import json
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import pickle
import albumentations as A
# import torch
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path


class DataHandler():
    def __init__(self, project_data_dict):
        self.project_data_dict = project_data_dict
        self.training_data = []
        self.inference_data = []
        self.base_path = project_data_dict["data"]["base_path"]
        self.labels = project_data_dict["dataset"]["labels"]
        
        # find all directories in the data folder
        self.positions = [d for d in Path(self.base_path + "/QuPath/data").iterdir() if d.is_dir()]
        
        # filter out non-directories
        self.positions = [pos for pos in self.positions if pos.is_dir()]
        
        pbar = tqdm(total=len(self.positions), desc='Loading data')
        for pos in self.positions:
            data = project_data_dict.copy()
            pbar.set_description(f'Loading data for {pos}')
            # get corresponding paths
            data_dir = pos / "server.json"
            
            # get meta information, e.g. image size
            with open(data_dir) as f:
                data_server = json.load(f)
            # merge
            data["metadata"] = data_server["metadata"]
            data["labels"] = self.labels
            file_id = data["metadata"]["name"]
            data["file_id"] = file_id
            data["image_name"] = file_id
            
            #open corresponding geojson file
            _file = f"{self.base_path}/Annotations/{file_id}.geojson"
            with open(_file) as f:
                data_geo = json.load(f)
            data["features"] = data_geo["features"]
            
            image_path = f"{self.base_path}\\Images\\{data['file_id']}"
                
            if len(data["features"]) > 0:
                data["path"] = image_path
                
                d = TrainImage(data)
                self.training_data.append(d)
                self.inference_data.append(d)
            else:
                self.data["path"] = image_path
                d = TrainImage(data)
                self.inference_data.append(d)

            pbar.update(1)
        pbar.close()

        with open("train_data.pkl", "wb") as td:
            pickle.dump(self.training_data, td)
            
        with open("inference_data.pkl", "wb") as td:
            pickle.dump(self.inference_data, td)

        self.train_valid_split()
        
    
    def train_valid_split(self, ratio=0.2):
        self.train_data, self.valid_data = train_test_split(self.training_data, test_size=ratio, random_state=42)
        print('split into training: '+ str(len(self.train_data)) + ', and validation: ' + str(len(self.valid_data)))
    
    def train_test_split(self, test_ratio=0.1, valid_ratio=0.2, random=42):    
        self.train_val_data, self.test_data = train_test_split(self.training_data, test_size=test_ratio, random_state=42)
        self.train_data, self.valid_data = train_test_split(self.train_val_data, test_size=valid_ratio, random_state=random)
        print('split into training: '+ str(len(self.train_data)) + ', valid: ' + str(len(self.valid_data)) + ', and testing: ' + str(len(self.test_data)))
        print("Training data:")
        for i, d in enumerate(self.test_data):
            print(f"{i}: {d.info_dict['file_id']}")
    
    def run_data_loader(self):
        """Create data loaders for training and validation datasets"""
        augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            
            # A.GaussNoise(var_limit=(0.005, 0.01), p=0.5),
            # A.RandomBrightnessContrast(p=0.25),
            # A.RandomGamma(p=0.25),
            # A.PixelDropout(p=1, per_channel=True),
        ])

        kernel_size = self.project_data_dict["dataset"]["kernel_size"]
        self.train_ds = ImageDataset(self.train_data, (kernel_size, kernel_size), augmentations=augmentations)
        self.valid_ds = ImageDataset(self.valid_data, (kernel_size, kernel_size), augmentations=None)

        self.train_dl = DataLoader(
            self.train_ds, 
            batch_size=self.project_data_dict["training"]["dataloader"]["batch_size"],
            shuffle=True, 
            num_workers=self.project_data_dict["training"]["dataloader"]["num_workers"],
            pin_memory=self.project_data_dict["training"]["dataloader"]["pin_memory"],
            drop_last=self.project_data_dict["training"]["dataloader"]["drop_last"],
            persistent_workers=self.project_data_dict["training"]["dataloader"]["persistent_workers"]
        )
        self.valid_dl = DataLoader(
            self.valid_ds, 
            batch_size=self.project_data_dict["validation"]["dataloader"]["batch_size"], 
            shuffle=False, 
            num_workers=self.project_data_dict["validation"]["dataloader"]["num_workers"], 
            pin_memory=self.project_data_dict["validation"]["dataloader"]["pin_memory"], 
            drop_last=self.project_data_dict["validation"]["dataloader"]["drop_last"], 
            persistent_workers=self.project_data_dict["validation"]["dataloader"]["persistent_workers"]
        )

    def show_example_batch(self):
        batch = next(iter(self.train_dl))
        image = batch[0].numpy()
        mask = batch[1].numpy()
        weights = batch[2].numpy()

        return image, mask, weights
