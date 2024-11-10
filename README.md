
# Campus Vision Challenge 2024

### Solution Overview

This repository contains the solution for the **Campus Vision Challenge**, where the task was to develop an image classification model to predict the name of university buildings based on a image.

The model was trained on a dataset containing images of various university buildings taken from different angles and lighting conditions. The model can predict one of the 10 university building names accurately.

Our goal was to create a buiding classifier with low computational cost. Thus, we trained 3 different image classifications models for this challenge: YOLOv11x-cls (ultralytics), swinv2-tiny-patch4-window8-256 (huggingface/microsoft), swinv2-small-w8_3rdparty_in1k-256px (mmcv).

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

We try three different models known for their high accuracy in classification tasks: YOLOv11x-cls (ultralytics), swinv2-tiny-patch4-window8-256 (huggingface/microsoft), swinv2-small-w8_3rdparty_in1k-256px (mmcv).

The challenge had normalization statistics with mean and standard deviation of the dataset. It is useful when you're training from scratch by effectiely preventing large difference in pixel values causing slow convergence. But for our models, we don't use those. Or also useful when you're fine tuning on a dataset very different from pre-train dataset. We instead use mean and standard deviation of the dataset the model was pre-trained on. Our models are already pretrained on millions of images and using a new normalization statistics might cause the pre-trained models' features and weight not reusable for our model. The weights are optimized for a different normalization statistic.

#### a. Data Preprocessing and Augmentation


To improve the generalization and robustness of the model, we applied data preprocessing techniques:

**Huggingface/microsft (swinv2-tiny)**: We normalize the images using the feature extractor's mean and std. We also apply random horizontal flips and randon resized crop for training data. Both training and validation data are resized to the the size of the images the model was pre-trained on.
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

For **mmcv (swinv2 small)** we only have to take care of the configuration file. 
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
<br />

**Yolov11x** takes care of data-preprocessing on its own during training.


#### b. Model Selection

**Yolov11x**
```python
model = YOLO('yolo11x-cls.pt')
```

**swinv2_tiny (huggingface)**
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

**swinv2_small (mmcv)**
Download the model and provide the model location in the config file.
```python
!wget https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-small-w8_3rdparty_in1k-256px_20220803-b01a4332.pth -P ./checkpoints # Download the model
cfg.load_from = './checkpoints/swinv2-small-w8_3rdparty_in1k-256px_20220803-b01a4332.pth'  # Provide the model location in the config file
```

#### c. Challenges

The first challenge was model selection. There are countless classifications models avaialble. We also had the option to create our own classifiaction model from scratch. But we don't have the computation power to try out what works and what doesn't work for building classification. We went onto the internet to find the best performing models with low computational cost. 

The second challenge was to find hardware to train the models. We used Kaggle to train our models. Kaggle gives out free GPU for 30hr/week: Tesla P100 GPU with 16GB VRAM and 30GB RAM. 

I built mmcv from source because of the CUDA version mismatch with kaggle. You can't directly install mmcv if you have cuda higher than 12.1 More instruction for running in inference section.

---
### **Part 2: Training/Validating the Models**

#### Training:

**Yolov11x**
```python
results = model.train(data='/kaggle/input/buildings/dataset/', epochs=45, imgsz=512, save=True, device=0, val=True, plots=True)
```


**swinv2_tiny (huggingface)**
We provide training arguments and create a trainer instance using those arguments.
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


**swinv2_small (mmcv)**
We train the model using the config file and train script
```python
buildings_config=f'./configs/swin_transformer_v2/swinv2-small-w8_buildings.py'
with open(buildings_config, 'w') as f:
    f.write(cfg.pretty_text)
!python tools/train.py {buildings_config}
```
<br />

**Yolov11x** and **swinv2_tiny** were also validated each epoch while traning. **swinv2_small** was only validated every 5 epoch after the first 20 epochs. 

#### Performance Metrics (Validation):

Here are the performance metrics of the final trained model on the **validation set**:

**yolox10-cls**
- **Accuracy**: 99.841%
- **Precision**: 0.99842
- **Recall**: 0.99841
- **F1 Score**: 0.99841

**swinv2-tiny (huggingface)**
- **Accuracy**: 99.577%
- **Precision**: 0.99581
- **Recall**: 0.99583
- **F1 Score**: 0.99581

**swinv2-small (mmcv)**
- **Accuracy**: 99.894%
- **Precision**: 0.99895
- **Recall**: 0.99894
- **F1 Score**: 0.9989

---

### **Part 3: Testing the Models**


The models with best validation metrics were saved for inference. We now test the best models on the test split of the dataset.

#### Model Evaluation (Testing)

Here are the performance metrics of the final trained model on the **test set**:

**yolox10-cls**
- **Accuracy**: 99.524%
- **Precision**: 0.99512
- **Recall**: 0.99536
- **F1 Score**: 0.99521

**swinv2-tiny (huggingface)**
- **Accuracy**: 99.577%
- **Precision**: 0.99582
- **Recall**: 0.99583
- **F1 Score**: 0.99582

**swinv2-small (mmcv)**
- **Accuracy**: 99.736%
- **Precision**: 0.99735
- **Recall**: 0.99742
- **F1 Score**: 0.99737
  
#### Confusion Matrix:
<img width="390" alt="yolo_confusion_matrix" src="https://raw.githubusercontent.com/pc942/cv_challenge_apt_4201/refs/heads/main/images/confusion_matrix_yolo.png"> 
<img width="390" alt="swinv2_tiny_confusion_matrix" src="https://raw.githubusercontent.com/pc942/cv_challenge_apt_4201/refs/heads/main/images/confusion_matrix_swinv2_tiny.png"> <br />
<img width="800" alt="yolox_results" src="https://raw.githubusercontent.com/pc942/cv_challenge_apt_4201/refs/heads/main/images/confusion_matrix_swinv2_small.png"> <br />

---

### **Part 4: Solution Files**

```markdown
## Solution Files

- **`yolo_inference.py`**: Script to get prediction results/metrics using yolov11x model.
- **`swinv2_small_inference.py`**: Script to get prediction results/metrics using swinv2_small model.
- **`swinv2_tiny_inference.py`**: Script to get prediction results/metrics using swinv2_tiny model.
- **`train/ultralytics_yolov11x.ipynb`**: Notebook to train yolov11x model on the dataset.
- **`train/mmdet_swimv2_small.ipynb`**: Notebook to train swinv2_small model on the dataset.
- **`train/huggingface_swinv2_tiny.ipynb`**: Notebook to train swinv2_tiny model on the dataset.
- **`predict.py`**: Script to load the trained model and make predictions on new images.
- **`create_dataset.ipynb`**: Notebook file to split the dataset into train/val/test split.
- **`requirements.txt`**: List of required libraries and dependencies.
```


#### Instructions to Run the Code for Inference

I created a new python virtual environment for inference so we would only need to install required packages for inference. The packages mentioned in the `requirement.txt` file might not be enough for training but only for inference. If anyone runs into problem during inference or training, please contact me through my email address mentioned in the bottom section. Link to the model files: https://drive.google.com/file/d/1xoQmGaSOGLqyvIDwyI5w0cT3dOUrKWs1/view?usp=sharing

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
  **For prediction:**
  
3. **Make Predictions**:
    To make predictions on new image dataset, use the following script:
    **yolov11x:**
    ```bash
    python yolo_inference.py --image_path <path_to_image_dataset> --model_path <path_to_model>
    ```
    Keep in mind, you can either give out an image or a dataset with **labels** (divided into subdirectories) to image_path. But when you are inferencing on a new dataset **without labels** (not divided into subdirectories). Use the following script:
    ```bash
    python yolo_inference.py --image_path <path_to_image_dataset> --model_path <path_to_model> --no_label
    ```
    <br />
    
   **swinv2_tiny (huggingface):**
    Hugging face uses `config.json` and `preprocessor_config.json` along with the model file in the same directory to load model and predict. Be sure to put them in the same directory. The download link to the model already has it in the same directory.
    ```bash
    python swinv2_tiny_inference.py --image_path <path_to_image_dataset> --model_folder_path <path_to_model_directory>
    ```
    Keep in mind, you can either give out an image or a dataset with **labels** (divided into subdirectories) to `image_path`. But when you are inferencing on a new dataset **without labels** (not divided into subdirectories). Use the following script:
    ```bash
    python swinv2_tiny_inference.py --image_path <path_to_image_dataset> --model_folder_path <path_to_model_directory> --no_label
    ```
    <br />
    
   **swinv2_small (mmcv):**
    MMCV uses `config.py` along with the model file in the same directory to load model and predict. Be sure to put them in the same directory. The download link to the model already has it in the same directory.
    ```bash
    python swinv2_small_inference.py --image_path <path_to_image_dataset> --model_folder_path <path_to_model_directory>
    ```
    Keep in mind, you can either give out an image or a dataset with **labels** (divided into subdirectories) to `image_path`. But when you are inferencing on a new dataset **without labels** (not divided into subdirectories). Use the following script:
    ```bash
    python swinv2_small_inference.py --image_path <path_to_image_dataset> --model_folder_path <path_to_model_directory> --no_label
    ```
 An example of running the script:
 <img width="800" alt="yolox_results" src="https://raw.githubusercontent.com/pc942/cv_challenge_apt_4201/refs/heads/main/images/example_run.png"> <br />
 **For training:**
       
   ```bash
   jupyter lab
   ```
   
3. **Preprocess the Dataset**:
    After step 2, run the preprocessing ipynb file (`create_dataset.ipynb`) to prepare the dataset (train/validation/test split):

4. **Train the Model**:
    Run the training ipynb files under the `training` directory. Each model has a different training ipynb file: `ultralytics_yolov11x.ipynb`, `mmdet_swimv2_small.ipynb`, `huggingface_swinv2_tiny.ipynb`.


---

### Conclusion

This solution demonstrates the effectiveness of using pre-trained yolov11 and swinv2 models for image classification tasks, particularly in predicting the names of university buildings. The performance metrics (accuracy, f1-score, etc.) can't improve much at this point.

Future work could instead include experimenting with more advanced models such as COCA for zero shot detection. Instead of retraining the model, we use the model as feature extractor. Now that the model has learnt to extract building features very accurately, we instead tell the model what other buildings look like. COCA uses image-text encoder decoder model to perform zero-shot detection. Without giving it images of, for example, Dogwood Hall, we would instead describe what it looks like. The COCA model could then predict whether the image is Dogwood hall or some other building without even a single picture of dogwood hall. 


### Team Members:

Piyush Chaudhary (pc942@msstate.edu)
Asahi Lama Sherpa (al2402@msstate.edu)
Sarthak Neupane (sn942@msstate.edu)
Pragyesh Poudel (pp895@msstate.edu)