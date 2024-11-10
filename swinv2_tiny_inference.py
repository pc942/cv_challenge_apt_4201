from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import gc
import torch
import os
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

# Create the parser
parser = argparse.ArgumentParser(description='(huggingface) Swinv2 tiny inference argparser')

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

model_path = Path(args.model_folder_path)
image_path = Path(args.image_path)

if (not os.path.isdir(model_path)):
    print('Please follow the README.md file and insert the folder path with config.json and preprocessor_config.json along with the model')
    exit()
if ('config.json' not in os.listdir(model_path)):
    print('Please follow the README.md file and insert the folder path with config.json and preprocessor_config.json along with the model')
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

#model.config.id2label give a dict {0: 'Butler Hall}, 1: 'Carpenter Hall', etc.....}
feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path)
model.to(device)


if is_dir:
    torchMetrics = Metrics()
    print('Predicting... ')

    ## NO LABELS
    if args.no_labels:
        for filename in os.listdir(image_path):
            filepath = image_path / Path(filename)
            img = Image.open(filepath)
            encoding = feature_extractor(img.convert("RGB"), return_tensors="pt")
            encoding.to(device)

            # forward pass
            with torch.no_grad():
                outputs = model(**encoding)
                logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            print(f"{str(filepath).split('/')[-1]}: {model.config.id2label[predicted_class_idx]}")
    ## LABELS
    else:
        torchMetrics = Metrics()
        for idx, folder in tqdm(model.config.id2label.items(), total=len(model.config.id2label.items()), desc='Buidings'):
            folder_path = image_path / Path(folder)
            for filename in tqdm(os.listdir(folder_path), desc='Files ', leave=False):
                filepath = folder_path / Path(filename)
                img = Image.open(filepath)
                encoding = feature_extractor(img.convert("RGB"), return_tensors="pt")
                encoding.to(device)
    
                # forward pass
                with torch.no_grad():
                    outputs = model(**encoding)
                    logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()
                torchMetrics.process_batch(predicted_class_idx, idx)
    
        print('\n\nPrecision, Recall and F1-score were weighted as in the instrcutions for the challenge')
        metrics = torchMetrics.get_metrics()
        print(metrics)
    
        #Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = sns.heatmap(metrics['cm'], annot=True, fmt="d", cmap="Blues", xticklabels=model.config.id2label.values(), yticklabels=model.config.id2label.values())
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Swinv2_tiny Confusion Matrix")
        plt.savefig('./confusion_matrix_swinv2_tiny.png')
        print('Confusion Matrix saved in ./confusion_matrix_swinv2_tiny.png')
else:
    img = Image.open(image_path)
    encoding = feature_extractor(img.convert("RGB"), return_tensors="pt")
    encoding.to(device)

    # forward pass
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    print(model.config.id2label[predicted_class_idx])