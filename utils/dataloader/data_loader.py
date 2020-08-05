from torch.utils.data import Dataset,random_split
from PIL import Image
import glob
import pickle 
import torch 
import numpy as np
import time

class EndGameDataset(Dataset):
    def __init__(self, dataset_paths):
        dataset_paths_pickle = open(dataset_paths,"rb")
        self.dataset_paths = pickle.load(dataset_paths_pickle)

    def __getitem__(self, idx):
        filepaths = self.dataset_paths[idx]
        fg_bg = self.get_input(filepaths["fg_bg"])
        bg = self.get_input(filepaths["bg"])
        fg_bg_mask = self.get_target(filepaths["fg_bg_mask"])
        dense_depth = self.get_target(filepaths["dense_depth"])
        return {"fg_bg":fg_bg, "bg":bg, "fg_bg_mask":fg_bg_mask, "dense_depth":dense_depth}

    def __len__(self):
        return len(self.dataset_paths)

    def get_input(self, filepath):
      return self.load_file(filepath)

    def get_target(self, filepath):
      target = self.load_file(filepath)
      # out = np.zeros(target.shape[0:2])
      target = target[:,:,np.newaxis]
      return target

    def load_file(self, img_path):
        img = Image.open(img_path)
        img_nd = np.array(img)
        if(np.max(img_nd)<1e-6):
          img_nd = img_nd
        else:
          img_nd = img_nd/np.max(img_nd)
        return img_nd

    def get_folder_file_paths(self, folderPath):
      file_paths = []
      for file in glob.glob(folderPath+"/**/*.jpg", recursive=True):
        file_paths.append(file)
      return file_paths

class EndGameSubset(Dataset):
    def __init__(self, subset, transform=None, target_transform=None):
        self.subset = subset
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        result = self.subset[index]
        if self.transform:
            result['fg_bg'] = self.transform["fg_bg"](result['fg_bg'])
            result['bg'] = self.transform["bg"](result['bg'])
        if self.target_transform:
          result['fg_bg_mask'] = self.target_transform["fg_bg_mask"](result['fg_bg_mask'])
          result['dense_depth'] = self.target_transform["dense_depth"](result['dense_depth'])
        return result

    def __len__(self):
        return len(self.subset)

def EndGameTrainTestDataSet(dataset_paths, train_split = 70, train_transforms=None, train_target_transforms=None, test_transforms=None, test_target_transforms=None):
  dataset = EndGameDataset(dataset_paths)
  train_len = len(dataset)*train_split//100
  test_len = len(dataset) - train_len 
  train_set, val_set = random_split(dataset, [train_len, test_len])
  train_dataset = EndGameSubset(train_set, transform=train_transforms, target_transform=train_target_transforms)
  test_dataset = EndGameSubset(val_set, transform=test_transforms, target_transform=test_target_transforms)

  return train_dataset, test_dataset