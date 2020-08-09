import pickle
import glob

def get_folder_file_paths(folderPath, pattern):
      """
      Filters files in the folder based on the pattern
      Returns file paths list
      """
      file_paths = []
      for file in glob.glob(folderPath+pattern, recursive=True):
        file_paths.append(file)
      return file_paths

def get_number(x):
    """
    Returns number(index) of the image from the filename
    This is used for sorting the files
    """
    filename = x.split("/")[-1]
    return int(filename.split(".")[0])


def serialize_filepaths_data(datasetPaths, dataPicklePath="./"):
    """
    Serializes dataset filepaths in the format required to run the model 
    Format :: [{"bg":"path/to/bg", "fg_bg":"path/to/fg_bg", "fg_bg_mask":"path/to/fg_bg_mask", "dense_depth":"path/to/dense_depth",}]
    """
    dataset_filepaths = []
    bg_filepaths = sorted(get_folder_file_paths(datasetPaths.get("bg"), "/*.jpg"), key=get_number)

    for bg in bg_filepaths:
        folder = str(get_number(bg))
        fg_bg_filepaths = sorted(get_folder_file_paths(datasetPaths.get("fg_bg"), "/"+folder+"/*.jpg"), key=get_number)
        fg_bg_mask_filepaths = sorted(get_folder_file_paths(datasetPaths.get("fg_bg_mask"), "/"+folder+"/*.jpg"), key=get_number)
        dense_depth_filepaths = sorted(get_folder_file_paths(datasetPaths.get("dense_depth"), "/"+folder+"/*.jpg"), key=get_number)
        for i in range(0, len(fg_bg_filepaths)):
            filePaths = {}
            filePaths["bg"] = bg
            filePaths["fg_bg"] = fg_bg_filepaths[i]
            filePaths["fg_bg_mask"] = fg_bg_mask_filepaths[i]
            filePaths["dense_depth"] = dense_depth_filepaths[i]
            dataset_filepaths.append(filePaths)

    pickle_out = open(dataPicklePath+"dataset_filepaths.pickle","wb")
    pickle.dump(dataset_filepaths, pickle_out)
    pickle_out.close()

def deserialize_filepaths_data(dataPicklePath='./'):
    """
    Deserializes the filepaths 
    """
    pickle_in = open(dataPicklePath+"dataset_filepaths.pickle","rb")
    deserialized_data = pickle.load(pickle_in)
    print(len(deserialized_data))
    return deserialized_data