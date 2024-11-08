# Campus Vision Challenge 2024

### Solution Overview

This repository contains the solution for the **Campus Vision Challenge**, where the task was to develop an image classification model to predict the name of university buildings based on a image.

The model was trained on a dataset containing images of various university buildings taken from different angles and lighting conditions. The model can predict one of the 10 university building names accurately.

Our goal was to create a buiding classifier with low computational cost. Thus, we trained 3 different image classifications models for this challenge: YOLOv11x-cls (ultralytics), swinv2-tiny-patch4-window8-256 (huggingface/microsoft), swinv2-small-w8_3rdparty_in1k-256px (mmdet).

### Problem Statement

We were tasked with classifying images of university buildings into one of 10 categories. The list of buildings includes:

- Butler Hall
- Carpenter Hall
- Lee Hall
- McCain Hall
- McCool Hall
- Old Main
- Simrall Hall
- Student Union
- Swalm Hall
- Walker Hall

### **Part 1: Dataset Creation** 

The dataset consists of images organized by building name into separate folders. Each image is in the folder with the name of the building it represents. 

#### Dataset Folder Structure

```bash
/dataset/
    ├── train/
    │   ├── Butler Hall/
    │   ├── Carpenter Hall/
    │   └── ...
    └── val/
    │   ├── Butler Hall/
    │   ├── Carpenter Hall/
    │   └── ...
    └── test/
    │   ├── Butler Hall/
    │   ├── Carpenter Hall/
    │   └── ...
    
```
 We use `create_dataset.ipynb` to create our train/test/val split in a different folder named `dataset`. A code snippet from the file:

 ```python
# Loop through each building folder in the data directory
for building_name in os.listdir(data_dir):
    building_path = os.path.join(data_dir, building_name)

    # Check if it's a directory and not a file
    if os.path.isdir(building_path):
        # Count the total number of images before dataset creation
        total_images_before = len([f for f in os.listdir(building_path) if os.path.isfile(os.path.join(building_path, f))])
        print(f"Building '{building_name}' - Before dataset creation: {total_images_before} images")

        # Make corresponding directories in the dataset folder
        os.makedirs(os.path.join(dataset_dir, 'train', building_name), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, 'val', building_name), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, 'test', building_name), exist_ok=True)

        # Get all image files in the building folder
        image_files = [f for f in os.listdir(building_path) if os.path.isfile(os.path.join(building_path, f))]
        
        # Split images into train, val, and test sets (70/15/15)
        train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)  # 0.5 * 0.3 = 0.15

        # Move files to the appropriate directories
        for file in train_files:
            shutil.copy(os.path.join(building_path, file), os.path.join(dataset_dir, 'train', building_name, file))
        
        for file in val_files:
            shutil.copy(os.path.join(building_path, file), os.path.join(dataset_dir, 'val', building_name, file))
        
        for file in test_files:
            shutil.copy(os.path.join
```

---

### **Part 2: Solution Approach & Challenges**

We try three different models known for their high accuracy in classification tasks: YOLOv11x-cls (ultralytics), swinv2-tiny-patch4-window8-256 (huggingface/microsoft), swinv2-small-w8_3rdparty_in1k-256px (mmdet).

The challenge had normalization statistics with mean and standard deviation of the dataset. It is useful when you're training from scratch by effectiely preventing large difference in pixel values causing slow convergence. But for our models, we don't use those. Or also useful when you're fine tuning on a dataset very different from pre-train dataset. We instead use mean and standard deviation of the dataset the model was pre-trained on. Our models are already pretrained on millions of images and using a new normalization statistics might cause the pre-trained models' features and weight not reusable for our model. The weights are optimized for a different normalization statistic.

#### a. Data Preprocessing and Augmentation


To improve the generalization and robustness of the model, we applied data preprocessing techniques:

Huggingface/microsft (swinv2-tiny)
```python
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
train_transforms = Compose(
        [
            RandomResizedCrop(feature_extractor.size['height']),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

val_transforms = Compose(
        [
            Resize(feature_extractor.size['height']),
            CenterCrop(feature_extractor.size['height']),
            ToTensor(),
            normalize,
        ]
    )
def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [
        val_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch
```

For mmdet (swinv2 small):
```python
cfg.model.head.num_classes = 10
cfg.data_preprocessor.num_classes = 10
cfg.load_from = './checkpoints/swinv2-small-w8_3rdparty_in1k-256px_20220803-b01a4332.pth'
cfg.work_dir = './work_dir'

cfg.dataset_type = 'CustomDataset'
cfg.train_dataloader.batch_size = batch_size
cfg.train_dataloader.num_workers = 2
cfg.train_dataloader.dataset.type = cfg.dataset_type

cfg.train_dataloader.dataset.data_root = '/kaggle/input/buildings/dataset/train'
del cfg.train_dataloader.dataset['split']



cfg.val_dataloader.batch_size = cfg.train_dataloader.batch_size
cfg.val_dataloader.num_workers = cfg.train_dataloader.num_workers
cfg.val_dataloader.dataset.data_root = '/kaggle/input/buildings/dataset/val'
cfg.val_dataloader.dataset.type = cfg.dataset_type
del cfg.val_dataloader.dataset['split']
```
We only have to take care of the configuration file. <br />

Yolov11 takes care of data-preprocessing on its own during training.


#### b. Model Selection

Yolov11x
```python
model = YOLO('yolo11x-cls.pt')
```

swinv2_tiny (huggingface)
```python
model_name_or_path = "microsoft/swinv2-tiny-patch4-window8-256"
model = AutoModelForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes = True, # fine tuning and change the num classes
)
```

swinv2_small (mmdet)
```python
!wget https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-small-w8_3rdparty_in1k-256px_20220803-b01a4332.pth -P ./checkpoints # Download the model
cfg.load_from = './checkpoints/swinv2-small-w8_3rdparty_in1k-256px_20220803-b01a4332.pth'  # Provide the model location in the config file
```

#### c. Challenges

The first challenge was model selection. There are countless classifications models avaialble. We also had the option to create our own classifiaction model from scratch. But we don't have the computation power to try out what works and what doesn't work for building classification. We went onto the internet to find the best performing models with low computational cost. 

The second challenge was to find hardware to train the models. We used Kaggle to train our models. Kaggle gives out free GPU for 30hr/week: Tesla P100 GPU with 16GB VRAM and 30GB RAM. 

I built mmdet from source because of the CUDA version mismatch with kaggle. But I think you can also directly install latest version of mmdet using mim. More instruction for running in inference section.

---
### **Part 2: Training/Validating the Models**

#### Training:

Yolov11x
```python
results = model.train(data='/kaggle/input/buildings/dataset/', epochs=45, imgsz=512, save=True, device=0, val=True, plots=True)
```

swinv2_tiny (huggingface)
```python
args = TrainingArguments(
    f"{model_name}-finetuned-buildings",
    remove_unused_columns=False,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=7e-4,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
#     metric_for_best_model="accuracy",
    metric_for_best_model="f1",
    push_to_hub=False,
)
trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)
train_results = trainer.train()
```

swinv2_small (mmdet)
```python
buildings_config=f'./configs/swin_transformer_v2/swinv2-small-w8_buildings.py'
with open(buildings_config, 'w') as f:
    f.write(cfg.pretty_text)
!python tools/train.py {buildings_config}
```

Yolov11x and swinv2_tiny were also validated each epoch while traning. swinv2_small was only validated every 5 epoch after the first 20 epochs.
<img width="800" alt="swinv2_tiny_table" src="https://github.com/pc942/cv_challenge_apt_4201/blob/main/images/swinv2_tiny_table.png"> <br />
<img width="800" alt="yolox_results" src="https://github.com/pc942/cv_challenge_apt_4201/blob/main/images/yolo_results.png"> <br />
<img width="800" alt="swinv2_small" src="https://github.com/user-attachments/assets/524a72c0-ce01-423d-b3ce-bc588de364e7">




#### Performance Metrics:

Here are the performance metrics of the final trained model on the validation set:

**yolox10-cls**
- **Accuracy**: 91.5%
- **Precision**: 0.90
- **Recall**: 0.88
- **F1 Score**: 0.89

**swinv2-tiny (huggingface)**
- **Accuracy**: 91.5%
- **Precision**: 0.90
- **Recall**: 0.88
- **F1 Score**: 0.89

**swinv2-small (mmdet)**
- **Accuracy**: 91.5%
- **Precision**: 0.90
- **Recall**: 0.88
- **F1 Score**: 0.89

##### Confision Matrix:

<img width="800" alt="swinv2_tiny_table" src="https://github.com/pc942/cv_challenge_apt_4201/blob/main/images/swinv2_tiny_table.png"> <br />
<img width="800" alt="yolox_results" src="https://github.com/pc942/cv_challenge_apt_4201/blob/main/images/yolo_results.png"> <br />
<img width="800" alt="swinv2_small" src="https://github.com/user-attachments/assets/524a72c0-ce01-423d-b3ce-bc588de364e7">

---

### **Part 3: Testing the Models**


The models with best validation metrics were saved for inferece. We now test the best models on the test split of the dataset.

#### Model Evaluation

Here are the performance metrics of the final trained model on the test set:

**yolox10-cls**
- **Accuracy**: 91.5%
- **Precision**: 0.90
- **Recall**: 0.88
- **F1 Score**: 0.89

**swinv2-tiny (huggingface)**
- **Accuracy**: 91.5%
- **Precision**: 0.90
- **Recall**: 0.88
- **F1 Score**: 0.89

**swinv2-small (mmdet)**
- **Accuracy**: 91.5%
- **Precision**: 0.90
- **Recall**: 0.88
- **F1 Score**: 0.89
  
#### Confusion Matrix
<img width="800" alt="swinv2_tiny_table" src="https://github.com/pc942/cv_challenge_apt_4201/blob/main/images/swinv2_tiny_table.png"> <br />
<img width="800" alt="yolox_results" src="https://github.com/pc942/cv_challenge_apt_4201/blob/main/images/yolo_results.png"> <br />
<img width="800" alt="swinv2_small" src="https://github.com/user-attachments/assets/524a72c0-ce01-423d-b3ce-bc588de364e7">
---

### **Part 4: Solution Files**

```markdown
## Solution Files

- **`model.py`**: Contains the implementation of the neural network architecture (ResNet18 with custom final layer).
- **`train.py`**: Script to train the model on the dataset.
- **`predict.py`**: Script to load the trained model and make predictions on new images.
- **`dataset_preprocessing.py`**: Script to preprocess and augment the dataset.
- **`requirements.txt`**: List of required libraries and dependencies.
```


#### Instructions to Run the Code for Inference

1. **Clone the Repository**:
    ```bash
    git clone <repo-link>
    cd <repo-name>
    ```

2. **Install Dependencies**:
    Install the required Python libraries using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

3. **Preprocess the Dataset**:
    Run the preprocessing script to prepare the dataset (train/validation split, augmentation):
    ```bash
    python dataset_preprocessing.py
    ```

4. **Train the Model**:
    Run the training script to train the model on the dataset:
    ```bash
    python train.py
    ```

    The model will save the best weights as `best_model.pth`.

5. **Make Predictions**:
    To make predictions on new images, use the following script:
    ```bash
    python predict.py --image_path <path_to_image>
    ```

---

### Conclusion

This solution demonstrates the effectiveness of using pre-trained yolov11 and swinv2 models for image classification tasks, particularly in predicting the names of university buildings. The accuracy and f1-score can't improve much at this point.

Future work could instead include experimenting with more advanced models such as COCA for zero shot detection. Instead of retraining the model, we use the model as feature extracter. Now that the model has learnt to extract building features very accurately, we instead tell the model what other buildings look like. COCA uses image-text encoder decoder model to perform zero-shot detection. Without giving it images of, for example, Dogwood Hall, we would instead describe what it looks like. The COCA model could then predict whether the image is Dogwood hall or some other building. 


### Team Members:

Piyush Chaudhary (pc942@msstate.edu)
Asahi Lama Sherpa (al2402@msstate.edu)
Sarthak Neupane (sn942@msstate.edu)
Pragyesh Poudel (pp895@msstate.edu)
