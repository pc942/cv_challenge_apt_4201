from mmpretrain import ImageClassificationInferencer
import gc
import torch
import os
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

# Create the parser
parser = argparse.ArgumentParser(description='(MMCV) Swinv2 small inference argparser')

# Define the arguments you expect
parser.add_argument('--image_path', type=str, help='Dataset path for testing', required=True)
parser.add_argument('--model_folder_path', type=str, help='Folder path for the model and configs', required=True)
parser.add_argument('--no_labels', action='store_true', help="Infer without any labels")

# Parse the command-line arguments
args = parser.parse_args()
is_dir = os.path.isdir(args.image_path)

# Access the parsed arguments
print(f"Image path: {args.image_path}")
print(f"Model path: {args.model_folder_path}")
print(f"No labels: {args.no_labels}")

gc.collect()
torch.cuda.empty_cache()

model_folder_path = Path(args.model_folder_path)
model_path = model_folder_path / Path('best.pth')
config_path = model_folder_path / Path('config.py')
image_path = Path(args.image_path)

if (not os.path.isdir(model_folder_path)):
    print('Please follow the README.md file and insert the folder path with config.py along with the model')
    exit()
if ('config.py' not in os.listdir(model_folder_path)):
    print('Please follow the README.md file and insert the folder path with config.py along with the model')
    exit()

class Metrics:
    def __init__(self):
        self.p = []
        self.t = []
        
    def process_batch(self, preds, targets):
        preds = np.asarray(preds).reshape(-1)  # Flatten preds to 1D array if necessary
        targets = np.asarray(targets).reshape(-1)  # Flatten targets to 1D array if necessary
        
        self.p = np.concatenate((self.p, preds), axis=0)
        self.t = np.concatenate((self.t, targets), axis=0)
    
    def get_metrics(self): 
        return {
            'accuracy': accuracy_score(self.t, self.p),
            'precision': precision_score(self.t, self.p, average='weighted'),
            'recall': recall_score(self.t, self.p, average='weighted'),
            'f1-score': f1_score(self.t, self.p, average='weighted'),
            'cm': confusion_matrix(self.t, self.p)
        }

device = torch.device('cuda') if torch.cuda.is_available() else (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))

#id2label give a dict {0: 'Butler Hall}, 1: 'Carpenter Hall', etc.....}
inferencer = ImageClassificationInferencer(str(config_path), pretrained=str(model_path), device=device)
id2label = {index: item for index, item in enumerate(inferencer.classes)}

if is_dir:
    torchMetrics = Metrics()
    print('Predicting... ')
    print('############################################################################################################################################')
    print('MMCV INFERENCE PROGRESS BAR CAN SHOW 0% PROGRESS WHILE IT IS INFERENCING')
    print('PLEASE DO NOT EXIT & WAIT FOR A WHILE')
    print('EACH INFERENCE BAR SHOWS PROGRESS OF A FOLDER CONTAINING A BUILDING IMAGES, 3 INFERENCE BAR MEANS IT IS IN THE THIRD FOLDER')
    print('############################################################################################################################################')
    
    ## NO LABELS
    if args.no_labels:
        file_list = [image_path / Path(filename) for filename in os.listdir(image_path)]
        results = inferencer(file_list)
        for i, result in enumerate(results):
            print(f"{str(file_list[i]).split('/')[-1]}: {result['pred_class']}")
    ## LABELS
    else:
        torchMetrics = Metrics()
        for idx, folder in tqdm(id2label.items(), total=len(id2label.items()), desc='Buidings'):
            folder_path = image_path / Path(folder)
            file_list = [folder_path / Path(filename) for filename in os.listdir(folder_path)]
            results = inferencer(file_list)
            for result in results:
                torchMetrics.process_batch(result['pred_label'], idx)
    
        print('\n\nPrecision, Recall and F1-score were weighted as in the instrcutions for the challenge')
        metrics = torchMetrics.get_metrics()
        print(metrics)
    
        #Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = sns.heatmap(metrics['cm'], annot=True, fmt="d", cmap="Blues", xticklabels=id2label.values(), yticklabels=id2label.values())
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Swinv2_small Confusion Matrix")
        plt.savefig('./confusion_matrix_swinv2_small.png')
        print('Confusion Matrix saved in ./confusion_matrix_swinv2_small.png')
else:
    results = inferencer(image_path)
    print(results[0]['pred_class'])