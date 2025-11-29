# %% [markdown]
# # Snow Pole Detection with Faster R-CNN
# 
# This notebook implements a Faster R-CNN model for detecting snow poles using PyTorch and torchvision.

# %%
# Install required packages if needed
# !pip install torch torchvision pillow matplotlib pycocotools

# %%
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.utils.data
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
import yaml

# %%
# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [markdown]
# ## Dataset Class
# 
# Create a custom dataset class to load images and YOLO format labels

# %%
class SnowPoleDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, image_folder, label_folder, transforms=None):
        """
        Args:
            data_root: Root directory containing data
            image_folder: Folder path relative to data_root for images (e.g., 'images/Train/train')
            label_folder: Folder path relative to data_root for labels (e.g., 'labels/Train/train')
            transforms: Optional transforms to be applied
        """
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, image_folder)
        self.label_dir = os.path.join(data_root, label_folder)
        self.transforms = transforms
        
        # Get all image files
        self.images = [f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.images.sort()
        
        print(f"Found {len(self.images)} images in {self.image_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        
        # Load corresponding label (YOLO format)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        boxes = []
        labels = []
        
        img_width, img_height = img.size
        
        # Parse YOLO format labels
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    # YOLO format: class_id center_x center_y width height (normalized)
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, cx, cy, w, h = map(float, parts)
                        
                        # Convert from YOLO format to [xmin, ymin, xmax, ymax]
                        xmin = (cx - w / 2) * img_width
                        ymin = (cy - h / 2) * img_height
                        xmax = (cx + w / 2) * img_width
                        ymax = (cy + h / 2) * img_height
                        
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(int(class_id) + 1)  # +1 because 0 is background
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Handle images with no boxes
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        if self.transforms:
            img = self.transforms(img)
        else:
            img = torchvision.transforms.ToTensor()(img)
        
        return img, target

# %% [markdown]
# ## Load Dataset Configuration

# %%
# Load data.yaml
# Get the script's directory and project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

# Try relative path first (for notebook), then absolute from project root
if os.path.exists('../../data.yaml'):
    data_yaml_path = '../../data.yaml'
elif os.path.exists('data.yaml'):
    data_yaml_path = 'data.yaml'
elif os.path.exists(os.path.join(project_root, 'data.yaml')):
    data_yaml_path = os.path.join(project_root, 'data.yaml')
else:
    raise FileNotFoundError("Cannot find data.yaml")

print(f"Script directory: {script_dir}")
print(f"Project root: {project_root}")
print(f"Loading data.yaml from: {data_yaml_path}")

with open(data_yaml_path, 'r') as f:
    data_config = yaml.safe_load(f)

# Get data root and paths - resolve relative to project root, not script location
data_root_from_yaml = data_config['path']
if os.path.isabs(data_root_from_yaml):
    data_root = data_root_from_yaml
else:
    # If path is relative, resolve it from project root
    data_root = os.path.abspath(os.path.join(project_root, data_root_from_yaml))

train_images = data_config['train']
val_images = data_config['val']
num_classes = data_config['nc'] + 1  # +1 for background. Num classes is simply 2 because we only predict snow poles and background


# Set subset size for testing (0.1 = 10%, 1.0 = 100%)
SUBSET_FRACTION = 1  # Change this to use more/less data

print(f"\nData root: {data_root}")
print(f"Number of classes (including background): {num_classes}")
print(f"Using {SUBSET_FRACTION * 100:.0f}% of the dataset for testing")

# Verify paths exist
train_img_path = os.path.join(data_root, train_images)
val_img_path = os.path.join(data_root, val_images)
print(f"\nVerifying data paths:")
print(f"Training images path: {train_img_path}")
print(f"  Exists: {os.path.exists(train_img_path)}")
print(f"Validation images path: {val_img_path}")
print(f"  Exists: {os.path.exists(val_img_path)}")

if not os.path.exists(train_img_path):
    raise FileNotFoundError(f"Training images directory not found: {train_img_path}")
if not os.path.exists(val_img_path):
    raise FileNotFoundError(f"Validation images directory not found: {val_img_path}")


# %%
# Create datasets
train_dataset_full = SnowPoleDataset(
    data_root=data_root,
    image_folder=train_images,
    label_folder='labels/Train/train'
)

val_dataset_full = SnowPoleDataset(
    data_root=data_root,
    image_folder=val_images,
    label_folder='labels/Validation/val'
)

# Create subsets if SUBSET_FRACTION < 1.0
if SUBSET_FRACTION < 1.0:
    train_size = int(len(train_dataset_full) * SUBSET_FRACTION)
    val_size = int(len(val_dataset_full) * SUBSET_FRACTION)
    
    # Create random subsets
    train_indices = torch.randperm(len(train_dataset_full))[:train_size].tolist()
    val_indices = torch.randperm(len(val_dataset_full))[:val_size].tolist()
    
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
    print(f"Full training samples: {len(train_dataset_full)} -> Using: {len(train_dataset)}")
    print(f"Full validation samples: {len(val_dataset_full)} -> Using: {len(val_dataset)}")
else:
    train_dataset = train_dataset_full
    val_dataset = val_dataset_full
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")


# %% [markdown]
# ## Visualize Sample Data

# %%
# Visualize a sample
def visualize_sample(dataset, idx):
    img, target = dataset[idx]
    
    # Convert tensor to numpy for visualization
    img_np = img.permute(1, 2, 0).numpy()
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_np)
    
    # Draw bounding boxes
    boxes = target['boxes'].numpy()
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
    
    ax.set_title(f"Sample {idx}: {len(boxes)} snow poles")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Show a few samples
for i in range(min(3, len(train_dataset))):
    visualize_sample(train_dataset, i)

# %% [markdown]
# ## Create Model

# %%
def get_model(num_classes):
    # Load a pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# Create model
model = get_model(num_classes)
model.to(device)

print(f"Model created with {num_classes} classes (including background)")

# %% [markdown]
# ## Data Loaders

# %%
def collate_fn(batch):
    return tuple(zip(*batch))

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

# %% [markdown]
# ## Training Setup

# %%
# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 10

# OPTION: Skip training and load pre-trained model
SKIP_TRAINING = True  # Set to False to retrain
MODEL_PATH = os.path.join(project_root, 'snow_pole_rcnn_final.pth')  # Path to trained model

# %% [markdown]
# ### Pytorch Faster RCNN Loss:
# When a Faster R-CNN model is in training mode, it returns a dictionary of losses (loss_dict), specifically:
# 
# -Classification Loss (loss_classifier): Cross-entropy loss for classifying region proposals as background or snow pole
# 
# -Box Regression Loss (loss_box_reg): Smooth L1 loss for refining bounding box coordinates
# 
# -RPN Classification Loss (loss_objectness): Binary cross-entropy for the Region Proposal Network to distinguish objects from background
# 
# -RPN Box Regression Loss (loss_rpn_box_reg): Smooth L1 loss for RPN's initial box proposals
# 
# **The total loss is the sum of all four losses, which is what gets backpropagated during training.**

# %% [markdown]
# ## Training Loop

# %%
if SKIP_TRAINING and os.path.exists(MODEL_PATH):
    print(f"\n{'='*50}")
    print(f"LOADING PRE-TRAINED MODEL: {MODEL_PATH}")
    print(f"{'='*50}\n")
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    train_losses = [0.0]  # Dummy value for compatibility
    
    print("✓ Model loaded successfully")
    print("Skipping training, proceeding to evaluation...\n")
else:
    # Training loop
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)
        
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
            if (i + 1) % 10 == 0:
                print(f"Batch [{i+1}/{len(train_loader)}], Loss: {losses.item():.4f}")
        
        # Update learning rate
        lr_scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"\nEpoch {epoch + 1} Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoints/rcnn_epoch_{epoch+1}.pth')
            print(f"Checkpoint saved at epoch {epoch + 1}")

    print("\nTraining completed!")

# %% [markdown]
# ## Plot Training Loss

# %%
# Plot training loss (only if training was performed)
if not SKIP_TRAINING and len(train_losses) > 1:
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_loss.png')
    plt.show()
else:
    print("Skipping training loss plot (model was loaded from checkpoint)")

# %% [markdown]
# ## Evaluation and Prediction

# %%
# Evaluation Metrics
from collections import defaultdict
import numpy as np

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def calculate_precision_recall_ap(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculate precision, recall, and AP for a single IoU threshold
    
    Args:
        predictions: List of (image_id, confidence, box) tuples
        ground_truths: Dict mapping image_id to list of ground truth boxes
        iou_threshold: IoU threshold for considering a detection as correct
    
    Returns:
        precision, recall, ap
    """
    # Sort predictions by confidence (descending)
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    
    # Count total ground truth boxes
    total_gt = sum(len(boxes) for boxes in ground_truths.values())
    
    if total_gt == 0:
        return 0.0, 0.0, 0.0
    
    # Track which ground truths have been matched
    gt_matched = {img_id: [False] * len(boxes) for img_id, boxes in ground_truths.items()}
    
    true_positives = []
    false_positives = []
    
    for img_id, conf, pred_box in predictions:
        if img_id not in ground_truths:
            false_positives.append(1)
            true_positives.append(0)
            continue
        
        gt_boxes = ground_truths[img_id]
        
        # Find best matching ground truth box
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_matched[img_id][gt_idx]:
                continue
            
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if match is good enough
        if best_iou >= iou_threshold and best_gt_idx != -1:
            gt_matched[img_id][best_gt_idx] = True
            true_positives.append(1)
            false_positives.append(0)
        else:
            false_positives.append(1)
            true_positives.append(0)
    
    # Compute cumulative sums
    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(false_positives)
    
    # Compute precision and recall
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recalls = tp_cumsum / total_gt
    
    # Compute AP using 11-point interpolation
    ap = 0.0
    for recall_threshold in np.linspace(0, 1, 11):
        precisions_above_threshold = precisions[recalls >= recall_threshold]
        if len(precisions_above_threshold) > 0:
            ap += np.max(precisions_above_threshold)
    ap /= 11.0
    
    # Final precision and recall
    final_precision = precisions[-1] if len(precisions) > 0 else 0.0
    final_recall = recalls[-1] if len(recalls) > 0 else 0.0
    
    return final_precision, final_recall, ap

def evaluate_model(model, dataset, conf_threshold=0.25):
    """
    Evaluate model on dataset and compute metrics
    
    Returns:
        dict with precision, recall, mAP@50, mAP@0.5:0.95
    """
    model.eval()
    
    all_predictions = []
    ground_truths = {}
    
    print("Collecting predictions...")
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            img, target = dataset[idx]
            
            # Get predictions
            prediction = model([img.to(device)])[0]
            
            # Filter by confidence
            keep = prediction['scores'] > conf_threshold
            pred_boxes = prediction['boxes'][keep].cpu().numpy()
            pred_scores = prediction['scores'][keep].cpu().numpy()
            
            # Store predictions
            for box, score in zip(pred_boxes, pred_scores):
                all_predictions.append((idx, score, box))
            
            # Store ground truths
            gt_boxes = target['boxes'].numpy()
            ground_truths[idx] = gt_boxes
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(dataset)} images")
    
    print(f"\nTotal predictions: {len(all_predictions)}")
    print(f"Total ground truth boxes: {sum(len(boxes) for boxes in ground_truths.values())}")
    
    # Calculate metrics at different IoU thresholds
    print("\nCalculating metrics...")
    
    # mAP@50
    precision_50, recall_50, ap_50 = calculate_precision_recall_ap(
        all_predictions, ground_truths, iou_threshold=0.5
    )
    
    # mAP@0.5:0.95 (average over IoU thresholds from 0.5 to 0.95 in steps of 0.05)
    aps = []
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    
    for iou_thresh in iou_thresholds:
        _, _, ap = calculate_precision_recall_ap(
            all_predictions, ground_truths, iou_threshold=iou_thresh
        )
        aps.append(ap)
    
    map_50_95 = np.mean(aps)
    
    results = {
        'Precision': precision_50,
        'Recall': recall_50,
        'mAP@50': ap_50,
        'mAP@0.5:0.95': map_50_95
    }
    
    return results

# Evaluation function
def predict_and_visualize(model, dataset, idx, conf_threshold=0.5):
    model.eval()
    
    img, target = dataset[idx]
    
    with torch.no_grad():
        prediction = model([img.to(device)])[0]
    
    # Filter predictions by confidence
    keep = prediction['scores'] > conf_threshold
    boxes = prediction['boxes'][keep].cpu().numpy()
    scores = prediction['scores'][keep].cpu().numpy()
    labels = prediction['labels'][keep].cpu().numpy()
    
    # Ground truth
    gt_boxes = target['boxes'].numpy()
    
    # Visualize
    img_np = img.permute(1, 2, 0).numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Ground truth
    ax1.imshow(img_np)
    for box in gt_boxes:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor='g', facecolor='none'
        )
        ax1.add_patch(rect)
    ax1.set_title(f'Ground Truth ({len(gt_boxes)} poles)')
    ax1.axis('off')
    
    # Predictions
    ax2.imshow(img_np)
    for box, score in zip(boxes, scores):
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax2.add_patch(rect)
        ax2.text(xmin, ymin - 5, f'{score:.2f}', 
                color='red', fontsize=10, weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax2.set_title(f'Predictions ({len(boxes)} poles, conf>{conf_threshold})')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return boxes, scores, labels

# %%
# Test on validation samples
for i in range(min(5, len(val_dataset))):
    boxes, scores, labels = predict_and_visualize(model, val_dataset, i, conf_threshold=0.5)

# %% [markdown]
# ### **Precision**
# Precision measures the accuracy of positive predictions:
# - **Formula:** `TP / (TP + FP)`
# - **Meaning:** Of all the snow poles the model detected, what percentage were actually real snow poles?
# - **High precision:** Model makes few false alarms (doesn't incorrectly detect non-poles as poles)
# 
# ### **Recall** 
# Recall measures how many actual snow poles were found:
# - **Formula:** `TP / (TP + FN)`
# - **Meaning:** Of all the actual snow poles in the images, what percentage did the model successfully detect?
# - **High recall:** Model finds most of the snow poles (doesn't miss many)
# 
# ### **mAP@50** (mean Average Precision at IoU=0.5)
# Average Precision calculated at an IoU (Intersection over Union) threshold of 0.5:
# - **Meaning:** A detection is considered correct if its bounding box overlaps ≥50% with the ground truth
# - **Standard metric:** Commonly used in object detection, balances precision and recall
# - **Range:** 0.0 to 1.0 (higher is better)
# 
# ### **mAP@0.5:0.95** (COCO metric)
# Average Precision calculated across multiple IoU thresholds (0.5, 0.55, 0.60, ..., 0.95):
# - **Meaning:** More strict metric that requires precise bounding box localization
# - **COCO standard:** Used in major computer vision competitions
# - **Harder to achieve:** Lower values are expected since it's more demanding
# - **Range:** 0.0 to 1.0 (higher is better)

# %%
# Evaluate on validation set for comparison
print("\n" + "="*50)
print("EVALUATING ON VALIDATION SET")
print("="*50 + "\n")

val_results = evaluate_model(model, val_dataset, conf_threshold=0.25)


print("\n" + "="*50)
print("VALIDATION SET RESULTS")
print("="*50)
print(f"Precision:       {val_results['Precision']:.4f}")
print(f"Recall:          {val_results['Recall']:.4f}")
print(f"mAP@50:          {val_results['mAP@50']:.4f}")
print(f"mAP@0.5:0.95:    {val_results['mAP@0.5:0.95']:.4f}")

# %% [markdown]
# ### Interpretation of results
# 
# We notice that the ground truth in many cases has 1 snow pole when actually 2 are visible. This could be due to labeling errors or occlusions. The model's performance metrics should be interpreted with this in mind, as they may not fully reflect the model's true detection capabilities.

# %%
# Save metrics to file
import json

metrics_dict = {
    'model': 'Faster R-CNN (ResNet50-FPN)',
    'dataset_fraction': SUBSET_FRACTION,
    'num_epochs': num_epochs,
    'validation_samples': len(val_dataset),
    'metrics': {
        'Precision': float(val_results['Precision']),
        'Recall': float(val_results['Recall']),
        'mAP@50': float(val_results['mAP@50']),
        'mAP@0.5:0.95': float(val_results['mAP@0.5:0.95'])
    },
    'final_training_loss': float(train_losses[-1])
}

with open('validation_metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=4)

print("Metrics saved to 'validation_metrics.json'")
print("\nMetrics Summary:")
print(json.dumps(metrics_dict, indent=2))

# %% [markdown]
# ## Save Final Model

# %%
# Save the final model
if not SKIP_TRAINING:
    torch.save(model.state_dict(), 'snow_pole_rcnn_final.pth')
    print("Final model saved as 'snow_pole_rcnn_final.pth'")

# %% [markdown]
# ## Generate Predictions on Test Set

# %%
print("\n" + "="*50)
print("GENERATING PREDICTIONS ON TEST SET")
print("="*50 + "\n")

# Load test images path from data.yaml
test_images = data_config['test']
test_img_path = os.path.join(data_root, test_images)

if not os.path.exists(test_img_path):
    print(f"Test images directory not found: {test_img_path}")
else:
    # Get all test images
    test_image_files = [f for f in os.listdir(test_img_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    test_image_files.sort()
    
    print(f"Found {len(test_image_files)} test images")
    
    # Create output directories
    pred_png_dir = 'predictions/PNG'
    pred_jpg_dir = 'predictions/jpg'
    os.makedirs(pred_png_dir, exist_ok=True)
    os.makedirs(pred_jpg_dir, exist_ok=True)
    
    model.eval()
    conf_threshold = 0.25  # Confidence threshold for predictions
    
    print(f"Generating predictions (confidence threshold: {conf_threshold})...\n")
    
    with torch.no_grad():
        for idx, img_name in enumerate(test_image_files):
            # Load image
            img_path = os.path.join(test_img_path, img_name)
            img = Image.open(img_path).convert("RGB")
            original_width, original_height = img.size
            
            # Convert to tensor
            img_tensor = torchvision.transforms.ToTensor()(img)
            
            # Get predictions
            prediction = model([img_tensor.to(device)])[0]
            
            # Filter by confidence
            keep = prediction['scores'] > conf_threshold
            pred_boxes = prediction['boxes'][keep].cpu().numpy()
            pred_scores = prediction['scores'][keep].cpu().numpy()
            pred_labels = prediction['labels'][keep].cpu().numpy()
            
            # Convert boxes to YOLO format (normalized center_x, center_y, width, height)
            yolo_predictions = []
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                xmin, ymin, xmax, ymax = box
                
                # Convert to YOLO format (normalized)
                center_x = ((xmin + xmax) / 2) / original_width
                center_y = ((ymin + ymax) / 2) / original_height
                width = (xmax - xmin) / original_width
                height = (ymax - ymin) / original_height
                
                # Class ID (subtract 1 to match YOLO format where background is not included)
                class_id = label - 1
                
                yolo_predictions.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f} {score:.6f}")
            
            # Determine output directory based on file extension
            file_ext = os.path.splitext(img_name)[1].lower()
            if file_ext == '.png':
                output_dir = pred_png_dir
            else:  # .jpg or .jpeg
                output_dir = pred_jpg_dir
            
            # Save predictions to txt file
            txt_filename = os.path.splitext(img_name)[0] + '.txt'
            output_path = os.path.join(output_dir, txt_filename)
            
            with open(output_path, 'w') as f:
                f.write('\n'.join(yolo_predictions))
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(test_image_files)} images")
    
    print(f"\n✓ Predictions saved to:")
    print(f"  - {pred_png_dir}/")
    print(f"  - {pred_jpg_dir}/")
    print(f"\nPrediction format: class_id center_x center_y width height confidence")

# %% [markdown]
# ## Load Model for Inference

# %%
# To load the model later:
# model = get_model(num_classes)
# model.load_state_dict(torch.load('snow_pole_rcnn_final.pth'))
# model.to(device)
# model.eval()


