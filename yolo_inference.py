from ultralytics import YOLO
import gc
import torch
import os
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

# Create the parser
parser = argparse.ArgumentParser(description='Yolo inference argparser')

# Define the arguments you expect
parser.add_argument('--image_path', type=str, help='Dataset path for testing', required=True)
parser.add_argument('--model_path', type=str, help='File path for the model', required=True)
parser.add_argument('--no_labels', action='store_true', help="Infer without any labels")

# Parse the command-line arguments
args = parser.parse_args()
is_dir = os.path.isdir(args.image_path)

# Access the parsed arguments
print(f"Image path: {args.image_path}")
print(f"Model path: {args.model_path}")
print(f"No labels: {args.no_labels}")

gc.collect()
torch.cuda.empty_cache()

model_path = Path(args.model_path)
image_path = Path(args.image_path)

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

#mode.name give a dict {0: 'Butler Hall}, 1: 'Carpenter Hall', etc.....}
model = YOLO(model_path)

if is_dir:
    torchMetrics = Metrics()
    print('Predicting... ')

    ## NO LABELS
    if args.no_labels:
        results = model.predict(image_path, imgsz=512, device=device, verbose=False)
        for result in results:
            print(f"{result.path.split('/')[-1]}: {model.names[result.probs.top1]}")

    ## LABELS
    else:
        for idx, folder in model.names.items():
            # folder = 'Butler Hall'
            results = model.predict(image_path/Path(folder), imgsz=512, device=device, verbose=False)
            for result in results:
                torchMetrics.process_batch(result.probs.top1, idx)
                # print(result.probs.top1, idx)
    
        print('Precision, Recall and F1-score were weighted as in the instrcutions for the challenge')
        metrics = torchMetrics.get_metrics()
        print(metrics)
    
        #Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = sns.heatmap(metrics['cm'], annot=True, fmt="d", cmap="Blues", xticklabels=model.names.values(), yticklabels=model.names.values())
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("YOLO Confusion Matrix")
        plt.savefig('./confusion_matrix_yolo.png')
        print('Confusion Matrix saved in ./confusion_matrix_yolo.png')
else:
    results = model.predict(image_path, imgsz=512, device=device, verbose=False)
    print(model.names[results[0].probs.top1])