class ExperimentLog():
    def __init__(self):
        self.path = "experiment-reports/"
        self.ext = ".log"
        self.model_name = ""
        self.drop_path_rate = ""
        self.epoch = ""
        self.image_size = ""
        self.learning_rate = ""
        self.bce_target_thresh = ""
        self.smoothing = ""
        self.mixup = ""
        self.cutmix = ""
        self.batch_size = ""
        self.loss_name = ""
        self.auto_augment = ""
        self.data_mean = ""
        self.data_std = ""
        self.optimizer = ""
        self.weight_decay = ""
        self.accuracy = ""
        self.precision = ""
        self.recall = ""
        self.early_stop = ""

    def write_experiment_details(self, filename):
        filename = filename
        f = open(self.path + filename + self.ext, "w")
        f.write("Model Name: "+self.model_name)
        f.write("\n")
        f.write("Drop Rate: "+self.drop_path_rate)
        f.write("\n")
        f.write("Epochs Trained: "+self.epoch)
        f.write("\n")
        f.write("Image Size: "+self.image_size)
        f.write("\n")
        f.write("Learning Rate: "+self.learning_rate)
        f.write("\n")
        f.write("BCE Thresh: "+self.bce_target_thresh)
        f.write("\n")
        f.write("Smoothing: "+self.smoothing)
        f.write("\n")
        f.write("Mixup: "+self.mixup)
        f.write("\n")
        f.write("Cutmix: "+self.cutmix)
        f.write("\n")
        f.write("Batch Size: "+self.batch_size)
        f.write("\n")
        f.write("Loss: "+self.loss_name)
        f.write("\n")
        f.write("Auto Augment: "+self.auto_augment)
        f.write("\n")
        f.write("Data Mean: "+self.data_mean)
        f.write("\n")
        f.write("Data Std: "+self.data_std)
        f.write("\n")
        f.write("Optimizer: "+self.optimizer)
        f.write("\n")
        f.write("Weight Decay: "+self.weight_decay)
        f.write("\n")
        f.write("Early Stop: "+self.early_stop)
        f.write("\n")
        f.write("Best Accuracy: "+self.accuracy)
        f.write("\n")
        f.write("Best Acc-Precision: "+self.precision)
        f.write("\n")
        f.write("Best Acc-Recall: "+self.recall)
        f.write("\n")