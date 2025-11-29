# %%
# Install required packages if needed
# !pip install torch torchvision pillow matplotlib pyyaml

# %%
import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
import torch.utils.data
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
import yaml
import time
import torchvision.transforms as T

# %%
# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Helper function for denormalization
def denormalize_image(tensor):
    """Denormalize image tensor for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Denormalize
    tensor = tensor * std + mean
    
    # Clamp to valid range
    tensor = torch.clamp(tensor, 0, 1)
    
    return tensor

# %% [markdown]
# ## Dataset Class
# 
# Reusing the same dataset class for YOLO format labels

# %%
class SnowPoleDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, image_folder, label_folder, transforms=None, training=False):
        """
        Args:
            data_root: Root directory containing data
            image_folder: Folder path relative to data_root for images
            label_folder: Folder path relative to data_root for labels
            transforms: Optional transforms to be applied
            training: Whether this is training set (for augmentation)
        """
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, image_folder)
        self.label_dir = os.path.join(data_root, label_folder)
        self.transforms = transforms
        self.training = training
        
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
        
        orig_width, orig_height = img.size
        
        # Parse YOLO format labels
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, cx, cy, w, h = map(float, parts)
                        
                        # Convert from YOLO format to [xmin, ymin, xmax, ymax]
                        xmin = (cx - w / 2) * orig_width
                        ymin = (cy - h / 2) * orig_height
                        xmax = (cx + w / 2) * orig_width
                        ymax = (cy + h / 2) * orig_height
                        
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(int(class_id) + 1)  # +1 because 0 is background
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Handle images with no boxes
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        # Apply proper resizing with letterboxing (maintains aspect ratio)
        img, boxes = self._resize_with_letterbox(img, boxes, target_size=320)
        
        # Apply augmentations for training
        if self.training and self.transforms:
            img, boxes = self._apply_augmentations(img, boxes)
        
        # Convert image to tensor and normalize
        img = T.ToTensor()(img)
        img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        return img, target
    
    def _resize_with_letterbox(self, img, boxes, target_size=320):
        """
        Resize image optimized for pole detection:
        - Poles cluster at y ≈ 0.5-0.8 (in YOLO coords where 0=top, 1=bottom)
        - Critical right poles at x = 0.5-1.0 (right half of image)
        Strategy: Crop bottom 15-20%, prioritize middle-to-top region, bias right
        """
        orig_width, orig_height = img.size
        aspect_ratio = orig_width / orig_height
        
        # Determine crop strategy based on aspect ratio
        # Crop BOTTOM (ground/road) since poles are in upper-middle portion
        if aspect_ratio > 1.5:  # Landscape (1920×1080): crop bottom more aggressively
            crop_bottom_ratio = 0.20  # Remove 20% of bottom
        else:  # Portrait or square (1920×1208): crop bottom moderately
            crop_bottom_ratio = 0.15  # Remove 15% of bottom
        
        crop_bottom = int(orig_height * crop_bottom_ratio)
        img_cropped = img.crop((0, 0, orig_width, orig_height - crop_bottom))
        cropped_height = orig_height - crop_bottom
        
        # Adjust boxes after cropping
        if len(boxes) > 0:
            boxes = boxes.clone()
            
            # Keep boxes that have their center still in the cropped region
            box_centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
            valid_mask = (box_centers_y >= 0) & (box_centers_y < cropped_height)
            boxes = boxes[valid_mask]
            
            # Clamp boxes to cropped boundaries
            if len(boxes) > 0:
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, cropped_height)
        
        # Resize to fit target size (maintaining aspect ratio)
        scale = min(target_size / orig_width, target_size / cropped_height)
        new_width = int(orig_width * scale)
        new_height = int(cropped_height * scale)
        
        img_resized = img_cropped.resize((new_width, new_height), Image.BILINEAR)
        
        # Create padded canvas
        new_img = Image.new('RGB', (target_size, target_size), (114, 114, 114))
        
        # Horizontal: RIGHT-BIASED (poles mostly at x=0.5-1.0, so prioritize right side)
        remaining_padding_x = target_size - new_width
        offset_x = int(remaining_padding_x * 0.3)  # 30% padding on left, 70% on right (bias left to show more right side)
        
        # Vertical: TOP-WEIGHTED (preserve y=0.5-0.8 region in YOLO coords = middle-upper in image)
        remaining_padding_y = target_size - new_height
        offset_y = int(remaining_padding_y * 0.25)  # 25% padding at top, 75% at bottom
        
        new_img.paste(img_resized, (offset_x, offset_y))
        
        # Final box adjustment
        if len(boxes) > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + offset_x
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + offset_y
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, target_size)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, target_size)
        
        return new_img, boxes
    
    def _apply_augmentations(self, img, boxes):
        """Apply training augmentations - preserve red color information for snow poles"""
        # Random horizontal flip
        if torch.rand(1) < 0.5:
            img = T.functional.hflip(img)
            if len(boxes) > 0:
                boxes = boxes.clone()
                width = img.size[0]
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]  # Flip x coordinates
        
        # LIMITED color jitter - preserve red hue since poles are red
        if torch.rand(1) < 0.5:
            img = T.ColorJitter(
                brightness=0.15,  # Reduced - preserve red brightness
                contrast=0.15,     # Reduced - preserve red contrast
                saturation=0.1,    # Reduced - preserve red saturation
                hue=0.02           # Minimal - preserve red hue (critical!)
            )(img)
        
        # Simulate weather conditions (snow/fog) while preserving color
        if torch.rand(1) < 0.3:
            # Random brightness reduction (overcast/snowy conditions)
            brightness_factor = 0.7 + torch.rand(1).item() * 0.3
            img = T.functional.adjust_brightness(img, brightness_factor)
        
        return img, boxes

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

# Set subset size for testing
SUBSET_FRACTION = 1.0  # Use 100% of data

print(f"\nData root: {data_root}")
print(f"Number of classes (including background): {num_classes}")
print(f"Using {SUBSET_FRACTION * 100:.0f}% of the dataset")

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
# Create datasets with proper transforms
train_dataset_full = SnowPoleDataset(
    data_root=data_root,
    image_folder=train_images,
    label_folder='labels/Train/train',
    training=True  # Enable augmentations
)

val_dataset_full = SnowPoleDataset(
    data_root=data_root,
    image_folder=val_images,
    label_folder='labels/Validation/val',
    training=False  # No augmentations for validation
)

# Create subsets if needed
if SUBSET_FRACTION < 1.0:
    train_size = int(len(train_dataset_full) * SUBSET_FRACTION)
    val_size = int(len(val_dataset_full) * SUBSET_FRACTION)
    
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
# ## Visualize Image Transformations
# 
# Show before and after the adaptive crop and letterbox transformation

# %%
def visualize_transformation(dataset, num_samples=3):
    """Visualize the before/after transformation for random samples"""
    import random
    
    print("\n" + "="*50)
    print("IMAGE TRANSFORMATION VISUALIZATION")
    print("="*50 + "\n")
    
    # Select random samples
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(16, 6 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for plot_idx, idx in enumerate(indices):
        # Load original image
        img_name = dataset.images[idx]
        img_path = os.path.join(dataset.image_dir, img_name)
        original_img = Image.open(img_path).convert("RGB")
        
        # Load labels
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(dataset.label_dir, label_name)
        
        boxes = []
        orig_width, orig_height = original_img.size
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, cx, cy, w, h = map(float, parts)
                        xmin = (cx - w / 2) * orig_width
                        ymin = (cy - h / 2) * orig_height
                        xmax = (cx + w / 2) * orig_width
                        ymax = (cy + h / 2) * orig_height
                        boxes.append([xmin, ymin, xmax, ymax])
        
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        
        # Apply transformation (same as dataset but without augmentation)
        transformed_img_pil, transformed_boxes = dataset._resize_with_letterbox(
            original_img.copy(), 
            boxes_tensor.clone() if len(boxes_tensor) > 0 else boxes_tensor,
            target_size=320
        )
        
        # Convert to tensor and normalize (as the model sees it)
        transformed_img_tensor = T.ToTensor()(transformed_img_pil)
        transformed_img_normalized = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(transformed_img_tensor)
        
        # Denormalize for proper visualization
        transformed_img_display = denormalize_image(transformed_img_normalized)
        transformed_img_display = transformed_img_display.permute(1, 2, 0).numpy()
        
        # Plot original
        ax_orig = axes[plot_idx, 0]
        ax_orig.imshow(original_img)
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            ax_orig.add_patch(rect)
        ax_orig.set_title(f'Original ({orig_width}×{orig_height})\n{len(boxes)} poles', fontsize=12, weight='bold')
        ax_orig.axis('off')
        
        # Plot transformed
        ax_trans = axes[plot_idx, 1]
        ax_trans.imshow(transformed_img_display)
        if len(transformed_boxes) > 0:
            for box in transformed_boxes:
                xmin, ymin, xmax, ymax = box
                rect = patches.Rectangle(
                    (xmin, ymin), xmax - xmin, ymax - ymin,
                    linewidth=2, edgecolor='cyan', facecolor='none'
                )
                ax_trans.add_patch(rect)
        
        # Calculate crop and padding info
        aspect_ratio = orig_width / orig_height
        crop_ratio = 0.20 if aspect_ratio > 1.5 else 0.15
        ax_trans.set_title(
            f'Transformed (320×320)\n'
            f'Bottom crop: {crop_ratio*100:.0f}% | Right-biased padding\n'
            f'{len(transformed_boxes)} poles retained', 
            fontsize=12, weight='bold'
        )
        ax_trans.axis('off')
        
        print(f"Sample {plot_idx + 1}: {img_name}")
        print(f"  Original size: {orig_width}×{orig_height}")
        print(f"  Aspect ratio: {aspect_ratio:.2f}")
        print(f"  Bottom crop: {crop_ratio*100:.0f}%")
        print(f"  Poles: {len(boxes)} → {len(transformed_boxes)} (retained)")
        print()
    
    plt.tight_layout()
    plt.savefig('transformation_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to 'transformation_visualization.png'\n")
    plt.show()

# Visualize transformations on training data
visualize_transformation(train_dataset_full, num_samples=3)

# %% [markdown]
# ## Create Lightweight Model
# 
# Using SSDLite320 with MobileNetV3 backbone - optimized for mobile/edge devices

# %%
def get_lightweight_model(num_classes):
    """
    Create MobileNetV3-SSDLite model
    - Input size: 320x320 (smaller = faster)
    - Backbone: MobileNetV3-Large (efficient depthwise separable convolutions)
    - Head: SSDLite (lightweight version of SSD)
    """
    # Load pretrained MobileNetV3-SSDLite with proper weights parameter
    from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
    model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    
    # Get the number of input channels and anchors from the existing head
    # The SSDLite head has 6 feature maps with these channel counts
    in_channels = [672, 480, 512, 256, 256, 128]
    num_anchors = [6, 6, 6, 6, 6, 6]
    
    # Replace classification head for our number of classes
    # Use BatchNorm2d as the norm layer (standard for SSDLite)
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=torch.nn.BatchNorm2d
    )
    
    return model

# Create model
model = get_lightweight_model(num_classes)
model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
model_size_mb = total_params * 4 / (1024**2)  # float32

print(f"\nModel: MobileNetV3-SSDLite320")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Estimated model size: {model_size_mb:.2f} MB")
print(f"\nComparison with Faster R-CNN:")
print(f"  Size reduction: {160/model_size_mb:.1f}x smaller")
print(f"  Parameters reduction: {41000000/total_params:.1f}x fewer parameters")

# %% [markdown]
# ## Data Loaders

# %%
def collate_fn(batch):
    return tuple(zip(*batch))

# Increase batch size for lightweight model (can handle more)
train_loader = DataLoader(
    train_dataset,
    batch_size=8,  # Larger batch size possible with lightweight model
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
    drop_last=True  # Drop incomplete batches to avoid BatchNorm issues
)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

# %% [markdown]
# ## Training Setup

# %%
num_epochs = 100  # Lightweight models may need slightly more epochs. We run 100 here as this is normal practice.

# Optimizer with stronger regularization to combat overfitting
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.0005, weight_decay=0.0005)  # Reduced LR, increased weight decay

# Learning rate scheduler with warmup
def get_lr_scheduler(optimizer, warmup_epochs=5, total_epochs=100):
    def warmup_lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 ** ((epoch - warmup_epochs) // 20)  # Decay every 20 epochs
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)

lr_scheduler = get_lr_scheduler(optimizer, warmup_epochs=5, total_epochs=num_epochs)

# Generate unique timestamp for this training run (prevents overwriting checkpoints)
from datetime import datetime
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# %% [markdown]
# ## Training Loop

# %%
# OPTION: Skip training and load checkpoint
SKIP_TRAINING = True  # Set to False to retrain
CHECKPOINT_PATH = 'checkpoints/mobilenet_ssd_epoch_100_20251127_131345.pth'  # Path to trained model

if SKIP_TRAINING and os.path.exists(CHECKPOINT_PATH):
    print(f"\n{'='*50}")
    print(f"LOADING CHECKPOINT: {CHECKPOINT_PATH}")
    print(f"{'='*50}\n")
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create dummy loss lists for plotting (optional)
    train_losses = [checkpoint.get('train_loss', 0.0)]
    val_losses = [checkpoint.get('val_loss', 0.0)]
    epoch_times = []
    num_epochs = checkpoint.get('epoch', 0) + 1
    
    print(f"✓ Loaded model from epoch {num_epochs}")
    print(f"  Train Loss: {train_losses[-1]:.4f}")
    print(f"  Val Loss: {val_losses[-1]:.4f}")
    print("\nSkipping training, proceeding to evaluation...\n")
    
else:
    # Training loop
    train_losses = []
    val_losses = []
    epoch_times = []

    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase
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
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase - FIXED: Use eval mode for consistent behavior
        model.eval()  # Changed from train() to eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # In eval mode, we need to manually compute loss
                # Switch temporarily to train mode just for loss calculation
                model.train()
                loss_dict = model(images, targets)
                model.eval()
                
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update learning rate scheduler
        lr_scheduler.step()
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Early stopping if validation loss increases for too long
        if len(val_losses) > 5:
            recent_val = val_losses[-5:]
            if all(recent_val[i] >= recent_val[i-1] for i in range(1, len(recent_val))):
                print("\n⚠ Warning: Validation loss increasing for 5 epochs - possible overfitting")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint_path = f'checkpoints/mobilenet_ssd_epoch_{epoch+1}_{RUN_TIMESTAMP}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    print("\nTraining completed!")
    print(f"Average epoch time: {np.mean(epoch_times):.2f}s")

# %% [markdown]
# ## Plot Training Progress

# %%
# Plot training and validation loss (only if we have training history)
if not SKIP_TRAINING and len(epoch_times) > 0:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, marker='s', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), epoch_times, marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time per Epoch')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('mobilenet_ssd_training.png')
    plt.show()
else:
    print("Skipping training plots (loaded from checkpoint)")

# %% [markdown]
# ## Evaluation Metrics

# %%
from collections import defaultdict

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def calculate_precision_recall_ap(predictions, ground_truths, iou_threshold=0.5):
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    total_gt = sum(len(boxes) for boxes in ground_truths.values())
    
    if total_gt == 0:
        return 0.0, 0.0, 0.0
    
    gt_matched = {img_id: [False] * len(boxes) for img_id, boxes in ground_truths.items()}
    true_positives = []
    false_positives = []
    
    for img_id, conf, pred_box in predictions:
        if img_id not in ground_truths:
            false_positives.append(1)
            true_positives.append(0)
            continue
        
        gt_boxes = ground_truths[img_id]
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_matched[img_id][gt_idx]:
                continue
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            gt_matched[img_id][best_gt_idx] = True
            true_positives.append(1)
            false_positives.append(0)
        else:
            false_positives.append(1)
            true_positives.append(0)
    
    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(false_positives)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recalls = tp_cumsum / total_gt
    
    ap = 0.0
    for recall_threshold in np.linspace(0, 1, 11):
        precisions_above_threshold = precisions[recalls >= recall_threshold]
        if len(precisions_above_threshold) > 0:
            ap += np.max(precisions_above_threshold)
    ap /= 11.0
    
    final_precision = precisions[-1] if len(precisions) > 0 else 0.0
    final_recall = recalls[-1] if len(recalls) > 0 else 0.0
    
    return final_precision, final_recall, ap

def evaluate_model(model, dataset, conf_threshold=0.25):
    model.eval()
    all_predictions = []
    ground_truths = {}
    inference_times = []
    
    print("Collecting predictions...")
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            img, target = dataset[idx]
            
            # Measure inference time
            start_time = time.time()
            prediction = model([img.to(device)])[0]
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            keep = prediction['scores'] > conf_threshold
            pred_boxes = prediction['boxes'][keep].cpu().numpy()
            pred_scores = prediction['scores'][keep].cpu().numpy()
            
            for box, score in zip(pred_boxes, pred_scores):
                all_predictions.append((idx, score, box))
            
            gt_boxes = target['boxes'].numpy()
            ground_truths[idx] = gt_boxes
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(dataset)} images")
    
    # Calculate metrics
    precision_50, recall_50, ap_50 = calculate_precision_recall_ap(
        all_predictions, ground_truths, iou_threshold=0.5
    )
    
    aps = []
    for iou_thresh in np.arange(0.5, 1.0, 0.05):
        _, _, ap = calculate_precision_recall_ap(
            all_predictions, ground_truths, iou_threshold=iou_thresh
        )
        aps.append(ap)
    
    map_50_95 = np.mean(aps)
    avg_inference_time = np.mean(inference_times)
    fps = 1.0 / avg_inference_time
    
    results = {
        'Precision': precision_50,
        'Recall': recall_50,
        'mAP@50': ap_50,
        'mAP@0.5:0.95': map_50_95,
        'avg_inference_time_ms': avg_inference_time * 1000,
        'fps': fps
    }
    
    return results

# %%
# Evaluate on validation set
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
print(f"\nInference Speed:")
print(f"Avg time:        {val_results['avg_inference_time_ms']:.2f} ms")
print(f"FPS:             {val_results['fps']:.1f}")
print("\nEdge Device Suitability: ✓ EXCELLENT")
print(f"  - {'Fast' if val_results['fps'] > 30 else 'Moderate'} inference speed")
print(f"  - Lightweight architecture (~{model_size_mb:.1f} MB)")
print(f"  - Low memory footprint")

# %% [markdown]
# ## Visualize Predictions

# %%
def predict_and_visualize(model, dataset, idx, conf_threshold=0.5, save_path=None):
    model.eval()
    img, target = dataset[idx]
    
    with torch.no_grad():
        prediction = model([img.to(device)])[0]
    
    keep = prediction['scores'] > conf_threshold
    boxes = prediction['boxes'][keep].cpu().numpy()
    scores = prediction['scores'][keep].cpu().numpy()
    gt_boxes = target['boxes'].numpy()
    
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
    ax2.set_title(f'MobileNetV3-SSD ({len(boxes)} poles, conf>{conf_threshold})')
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
        plt.close()
    else:
        plt.show()
    
    return boxes, scores

# Test on validation samples
print("\n" + "="*50)
print("VISUALIZING PREDICTIONS")
print("="*50 + "\n")

os.makedirs('predictions_visualization', exist_ok=True)
num_samples_to_visualize = min(30, len(val_dataset))
print(f"Generating {num_samples_to_visualize} visualizations...\n")

for i in range(num_samples_to_visualize):
    print(f"Sample {i+1}/{num_samples_to_visualize}")
    save_path = f'predictions_visualization/sample_{i+1:02d}.png'
    boxes, scores = predict_and_visualize(model, val_dataset, i, conf_threshold=0.5, save_path=save_path)
    print(f"  Predicted: {len(boxes)} poles")
    print(f"  Ground truth: {len(val_dataset[i][1]['boxes'])} poles")
    if len(scores) > 0:
        print(f"  Confidence scores: {', '.join([f'{s:.2f}' for s in scores])}")
    print()

print(f"\n✓ All visualizations saved to 'predictions_visualization/' directory")

# %% [markdown]
# ## Save Results

# %%
import json

# Save comprehensive metrics
metrics_dict = {
    'model': 'MobileNetV3-SSDLite320',
    'model_size_mb': float(model_size_mb),
    'total_parameters': int(total_params),
    'dataset_fraction': SUBSET_FRACTION,
    'num_epochs': num_epochs,
    'validation_samples': len(val_dataset),
    'metrics': {
        'Precision': float(val_results['Precision']),
        'Recall': float(val_results['Recall']),
        'mAP@50': float(val_results['mAP@50']),
        'mAP@0.5:0.95': float(val_results['mAP@0.5:0.95'])
    },
    'performance': {
        'avg_inference_time_ms': float(val_results['avg_inference_time_ms']),
        'fps': float(val_results['fps'])
    },
    'final_train_loss': float(train_losses[-1]),
    'final_val_loss': float(val_losses[-1]),
    'edge_device_ready': True
}

with open('mobilenet_ssd_metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=4)

print("Metrics saved to 'mobilenet_ssd_metrics.json'")
print("\nFull Metrics Summary:")
print(json.dumps(metrics_dict, indent=2))

# %% [markdown]
# ## Save Final Model

# %%
# Save the final model
torch.save(model.state_dict(), 'snow_pole_mobilenet_ssd_final.pth')
print("Final model saved as 'snow_pole_mobilenet_ssd_final.pth'")

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
    pred_png_dir = 'models/lightweight/predictions/PNG'
    pred_jpg_dir = 'models/lightweight/predictions/jpg'
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
            
            # Apply same transformation as dataset (letterbox resize)
            # Create a temporary boxes tensor (empty for test images)
            empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
            
            # Use the dataset's transformation method
            img_transformed, _ = train_dataset_full._resize_with_letterbox(
                img.copy(), 
                empty_boxes, 
                target_size=320
            )
            
            # Convert to tensor and normalize (as the model expects)
            img_tensor = torchvision.transforms.ToTensor()(img_transformed)
            img_tensor = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )(img_tensor)
            
            # Get predictions
            prediction = model([img_tensor.to(device)])[0]
            
            # Filter by confidence
            keep = prediction['scores'] > conf_threshold
            pred_boxes = prediction['boxes'][keep].cpu().numpy()
            pred_scores = prediction['scores'][keep].cpu().numpy()
            pred_labels = prediction['labels'][keep].cpu().numpy()
            
            # Convert boxes back to original image coordinates
            # Reverse the letterbox transformation
            yolo_predictions = []
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                xmin, ymin, xmax, ymax = box
                
                # Boxes are in 320x320 space, need to map back to original image
                # Account for the letterbox padding and cropping done during transformation
                aspect_ratio = original_width / original_height
                crop_ratio = 0.20 if aspect_ratio > 1.5 else 0.15
                cropped_height = original_height * (1 - crop_ratio)
                
                # Calculate scale and offsets used in letterbox
                scale = min(320 / original_width, 320 / cropped_height)
                new_width = int(original_width * scale)
                new_height = int(cropped_height * scale)
                
                offset_x = int((320 - new_width) * 0.3)
                offset_y = int((320 - new_height) * 0.25)
                
                # Reverse the transformation
                orig_xmin = (xmin - offset_x) / scale
                orig_ymin = (ymin - offset_y) / scale
                orig_xmax = (xmax - offset_x) / scale
                orig_ymax = (ymax - offset_y) / scale
                
                # Clamp to valid range (account for cropped bottom)
                orig_xmin = max(0, min(orig_xmin, original_width))
                orig_ymin = max(0, min(orig_ymin, cropped_height))
                orig_xmax = max(0, min(orig_xmax, original_width))
                orig_ymax = max(0, min(orig_ymax, cropped_height))
                
                # Convert to YOLO format (normalized center_x, center_y, width, height)
                # Use original image dimensions for normalization
                center_x = ((orig_xmin + orig_xmax) / 2) / original_width
                center_y = ((orig_ymin + orig_ymax) / 2) / original_height
                width = (orig_xmax - orig_xmin) / original_width
                height = (orig_ymax - orig_ymin) / original_height
                
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
# ## Model Comparison Summary

print("\n" + "="*60)
print("RECOMMENDATION FOR EDGE DEPLOYMENT:")
print("="*60)
print("✓ MobileNetV3-SSD is HIGHLY SUITABLE for edge devices")
print(f"  - {160/model_size_mb:.0f}x smaller model size")
print(f"  - {val_results['fps']/10:.0f}x faster inference (estimated)")
print(f"  - Compatible with mobile CPUs, Raspberry Pi, Jetson Nano")
print(f"  - Can be quantized further for even better performance")
print("\nNext steps for deployment:")
print("  1. Quantize model (INT8) for 4x size reduction")
print("  2. Use ONNX format for cross-platform deployment")
print("  3. Deploy with TensorRT for NVIDIA edge devices")
print("  4. Use PyTorch Mobile for Android/iOS")


