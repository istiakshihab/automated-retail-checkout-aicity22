import shutil
import os
import numpy as np
import argparse
# get dirs

def seperate_into_folder(source_path) : 
    datapath = source_path ## dataset/bgr_images
    dirpath, dirs, filenames = next(os.walk(datapath))

    for file in filenames:
        class_label = file.split("_")[0] 
        path_to_save = source_path.split("/")[0]+"/train/"+class_label ## dataset/train/001
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        shutil.copy(datapath+"/"+file, path_to_save)



def get_files_from_folder(path):

    files = os.listdir(path)
    return np.asarray(files)

def split_traindata(source_path, train_ratio):
    # get dirs
    datapath = source_path.split("/")[0]
    _, dirs, _ = next(os.walk(datapath+"/train"))

    # calculates how many train data per class
    data_counter_per_class = np.zeros((len(dirs)))
    for i in range(len(dirs)):
        path = os.path.join(datapath+"/train", dirs[i])
        files = get_files_from_folder(path)
        data_counter_per_class[i] = len(files)
    test_counter = np.round(data_counter_per_class * (1 - train_ratio))

    # transfers files
    for i in range(len(dirs)):
        path_to_original = os.path.join(datapath+"/train", dirs[i])
        path_to_save = os.path.join(datapath+"/valid", dirs[i])

        #creates dir
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        files = get_files_from_folder(path_to_original)
        # moves data
        for j in range(int(test_counter[i])):
            dst = os.path.join(path_to_save, files[j])
            src = os.path.join(path_to_original, files[j])
            shutil.move(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train test split.")
    parser.add_argument("--data_dir", required=True, help="The background removed image data folder on disk.")
    parser.add_argument("--train_file_ratio", required=True, help="Ratio of training file")
    args = parser.parse_args()
    seperate_into_folder(args.data_dir)
    split_traindata(args.data_dir, args.train_file_ratio)
