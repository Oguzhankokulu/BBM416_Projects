# %% [markdown]
#  # BBM418 Fourth Assignment - Final "Magic" Version
# 
#  # Name: Ahmet Oğuzhan Kökülü
# 
#  # Student ID: 2220356053

# %%
import re
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import resnet18, ResNet18_Weights
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torchvision.ops as ops 

# Install torchmetrics if missing
try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "torchmetrics"])
    from torchmetrics.detection.mean_ap import MeanAveragePrecision


# %% [markdown]
#  ## 1. Dataset (Robust Augmentation)

# %%
class PennFudanDataset(Dataset):
    def __init__(self, root_dir, split='train', transforms=None, augment=False):
        self.root_dir = root_dir
        self.transforms = transforms
        self.augment = augment 
        
        split_path = os.path.join(root_dir, "splits/"+split+".txt")
        # Handle line endings safely
        with open(split_path, "r") as split_file:
            content = split_file.read()
            self.images = content.splitlines() if "\n" in content else content.split()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx].split(".")[0]
        image_path = os.path.join(self.root_dir, "Pedestrian/"+img_name+".png")
        annotation_path = os.path.join(self.root_dir, "Pedestrian/"+img_name+".txt")

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # Parse Annotations
        bounding_boxes = []
        with open(annotation_path) as ann:
            for line in ann:
                if "Bounding box" in line:
                    bounding_box_coor = re.findall(r'\d+', line.split(":")[1])
                    bounding_boxes.append([int(x) for x in bounding_box_coor])

        bounding_boxes = torch.as_tensor(bounding_boxes, dtype=torch.float32)

        # --- MANUAL AUGMENTATION ---
        if self.augment:
            # 1. Random Scale/Crop (Zoom)
            if random.random() > 0.5:
                scale = random.uniform(0.6, 1.0)
                new_w, new_h = int(width * scale), int(height * scale)
                x_off = random.randint(0, width - new_w)
                y_off = random.randint(0, height - new_h)
                
                image = image.crop((x_off, y_off, x_off + new_w, y_off + new_h))
                
                if len(bounding_boxes) > 0:
                    bounding_boxes[:, 0] -= x_off
                    bounding_boxes[:, 1] -= y_off
                    bounding_boxes[:, 2] -= x_off
                    bounding_boxes[:, 3] -= y_off
                    bounding_boxes[:, 0].clamp_(0, new_w)
                    bounding_boxes[:, 1].clamp_(0, new_h)
                    bounding_boxes[:, 2].clamp_(0, new_w)
                    bounding_boxes[:, 3].clamp_(0, new_h)
                    
                    keep = (bounding_boxes[:, 2] > bounding_boxes[:, 0] + 5) & \
                           (bounding_boxes[:, 3] > bounding_boxes[:, 1] + 5)
                    bounding_boxes = bounding_boxes[keep]
                
                width, height = new_w, new_h

            # 2. Horizontal Flip
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                if len(bounding_boxes) > 0:
                    new_xmin = width - bounding_boxes[:, 2]
                    new_xmax = width - bounding_boxes[:, 0]
                    bounding_boxes[:, 0] = new_xmin
                    bounding_boxes[:, 2] = new_xmax

        # Normalize to [cx, cy, w, h] 0-1
        coordinates = []
        for box in bounding_boxes:
            xmin, ymin, xmax, ymax = box
            box_w = xmax - xmin
            box_h = ymax - ymin
            cx = xmin + box_w / 2
            cy = ymin + box_h / 2
            coordinates.append([cx/width, cy/height, box_w/width, box_h/height])

        box_tensor = torch.as_tensor(coordinates, dtype=torch.float32)
        if len(box_tensor) == 0:
            box_tensor = torch.zeros((0, 4), dtype=torch.float32)

        target = {}
        target["boxes"] = box_tensor
        target["labels"] = torch.ones(len(coordinates), dtype=torch.int64) # Class 1
        target["image_id"] = torch.tensor([idx])
        target["orig_size"] = torch.tensor([height, width]) 

        if self.transforms is not None:
            image = self.transforms(image)
        
        return image, target


# %% [markdown]
#  ## 2. Transforms & Loaders

# %%
stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

train_transform = T.Compose([
    T.Resize((512, 512)),
    T.RandomApply([T.ColorJitter(0.5, 0.5, 0.5, 0.1)], p=0.8),
    T.RandomGrayscale(p=0.2),
    T.RandomApply([T.GaussianBlur((5, 9))], p=0.5),
    T.ToTensor(),
    T.Normalize(*stats)
])

val_transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(*stats)
])

def collat_func(batch):
    images, targets = list(zip(*batch))
    images = torch.stack(images)
    return images, targets

root_path = "ignore" 
train_dataset = PennFudanDataset(root_path, split="train", transforms=train_transform, augment=True)
val_dataset = PennFudanDataset(root_path, split="val", transforms=val_transform, augment=False)
test_dataset = PennFudanDataset(root_path, split="test", transforms=val_transform, augment=False)

train_dloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collat_func)
val_dloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collat_func)
test_dloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collat_func)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")


# %% [markdown]
#  ## 3. TinyDETR with Magic Fix (Frozen BN)

# %%
class TinyDETR(nn.Module):
    def __init__(self, num_queries, num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers):
        super().__init__()

        # ResNet18 Backbone
        raw_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Dilation Fix: 16x16 -> 32x32 feature map
        raw_model.layer4[0].conv1.stride = (1, 1)
        raw_model.layer4[0].downsample[0].stride = (1, 1)
        
        self.backbone = nn.Sequential(*list(raw_model.children())[:-2])
        backbone_channels = 512
        
        # Unfreeze weights
        for param in self.backbone.parameters():
            param.requires_grad = True

        self.conv = nn.Conv2d(backbone_channels, hidden_dim, kernel_size=1)
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        
        # Better initialization
        self.query_pos = nn.Parameter(torch.zeros(num_queries, hidden_dim))
        nn.init.xavier_uniform_(self.query_pos)
        
        self.class_head = nn.Linear(hidden_dim, num_classes+1)
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    # --- MAGIC FIX: Force BN to Eval Mode ---
    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.backbone.train()
            # Freeze BN stats
            for module in self.backbone.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()

    def encode_position(self, d_model, height, width, device):
        x_pos = torch.arange(width, dtype=torch.float32, device=device)
        y_pos = torch.arange(height, dtype=torch.float32, device=device)
        dim_t = torch.arange(0, d_model//2, 2, dtype=torch.float32, device=device)
        div_term = torch.exp(dim_t * (-torch.log(torch.tensor(10000.0)) / (d_model // 2)))

        poen_x = torch.zeros(width, d_model//2, device=device)
        poen_x[:, 0::2] = torch.sin(x_pos.unsqueeze(1) * div_term)
        poen_x[:, 1::2] = torch.cos(x_pos.unsqueeze(1) * div_term)
        
        poen_y = torch.zeros(height, d_model // 2, device=device)
        poen_y[:, 0::2] = torch.sin(y_pos.unsqueeze(1) * div_term)
        poen_y[:, 1::2] = torch.cos(y_pos.unsqueeze(1) * div_term)

        poen_x = poen_x.unsqueeze(0).repeat(height, 1, 1)
        poen_y = poen_y.unsqueeze(1).repeat(1, width, 1)
        
        poen = torch.cat([poen_x, poen_y], dim=2)
        return poen.permute(2, 0, 1).unsqueeze(0)
        
    def forward(self, x):
        # Safety for BN
        for module in self.backbone.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

        features = self.backbone(x)
        h = self.conv(features) 
        B, C, H, W = h.shape
        pos_enc = self.encode_position(C, H, W, h.device)
        
        src = (h + pos_enc).flatten(2).permute(2, 0, 1)
        query_embed = self.query_pos.unsqueeze(1).repeat(1, B, 1)

        hs = self.transformer(src, query_embed)
        hs = hs.permute(1, 0, 2)
        
        return {'pred_logits': self.class_head(hs), 'pred_boxes': self.bbox_head(hs).sigmoid()}


# %% [markdown]
#  ## 4. Matcher & Loss

# %%
def box_c_to_xy(x):
    cx, cy, w, h = x.unbind(-1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim=-1)

def generalized_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]) 
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]) 
    lt = torch.max(boxes1[:, :2].unsqueeze(1), boxes2[:, :2].unsqueeze(0))
    rb = torch.min(boxes1[:, 2:].unsqueeze(1), boxes2[:, 2:].unsqueeze(0))
    wh = (rb - lt).clamp(min=0) 
    inter = wh[:, :, 0] * wh[:, :, 1] 
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter
    iou = inter / union
    lt_c = torch.min(boxes1[:, :2].unsqueeze(1), boxes2[:, :2].unsqueeze(0))
    rb_c = torch.max(boxes1[:, 2:].unsqueeze(1), boxes2[:, 2:].unsqueeze(0))
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, :, 0] * wh_c[:, :, 1]
    return iou - ((area_c - union) / (area_c + 1e-6))

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.w_class, self.w_bbox, self.w_giou = cost_class, cost_bbox, cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs = outputs["pred_logits"].shape[0]
        indices = []
        for i in range(bs):
            probs = outputs["pred_logits"][i].softmax(-1)
            pred_boxes = outputs["pred_boxes"][i]
            tgt_ids = targets[i]["labels"]
            tgt_boxes = targets[i]["boxes"]
            cost_class = -probs[:, tgt_ids]
            cost_bbox = torch.cdist(pred_boxes, tgt_boxes, p=1)
            cost_giou = -generalized_iou(box_c_to_xy(pred_boxes), box_c_to_xy(tgt_boxes))
            C = self.w_bbox * cost_bbox + self.w_class * cost_class + self.w_giou * cost_giou
            row, col = linear_sum_assignment(C.cpu().numpy())
            indices.append((torch.as_tensor(row, dtype=torch.int64), torch.as_tensor(col, dtype=torch.int64)))
        return indices

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs['pred_logits'].device)
        num_boxes = torch.clamp(num_boxes, min=1).item()
        
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.zeros(src_logits.shape[:2], dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').sum() / num_boxes
        loss_giou = (1 - generalized_iou(box_c_to_xy(src_boxes), box_c_to_xy(target_boxes)).diag()).sum() / num_boxes

        return {'loss_ce': loss_ce, 'loss_bbox': loss_bbox, 'loss_giou': loss_giou}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


# %% [markdown]
#  ## 5. Training with Integrated Evaluation

# %%
BATCH_SIZE = 4
EPOCHS = 100 
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 1      
NUM_QUERIES = 20  # Reduced to 20
HIDDEN_DIM = 256     

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Device: {device}")

model = TinyDETR(NUM_QUERIES, NUM_CLASSES, HIDDEN_DIM, 8, 2, 2).to(device)
matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
criterion = SetCriterion(NUM_CLASSES, matcher, {'loss_ce': 2.0, 'loss_bbox': 5, 'loss_giou': 2}, eos_coef=0.1).to(device)

param_dicts = [
    {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": 1e-5},
    {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad], "lr": 1e-4},
]
optimizer = optim.AdamW(param_dicts, weight_decay=WEIGHT_DECAY)

# Initialize Metric (Pedestrian is effectively class 0 for single-class metrics)
metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True).to(device)

loss_history = []
map_history = []

print("Starting Training...")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for images, targets in train_dloader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        loss = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_dloader)
    loss_history.append(avg_loss)

    # --- VALIDATION WITH CORRECT mAP CALCULATION ---
    if (epoch + 1) % 5 == 0: 
        model.eval()
        metric.reset()
        with torch.no_grad():
            for images, targets in val_dloader:
                images = images.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model(images)

                pred_list = []
                target_list = []
                for i in range(len(images)):
                    probs = outputs['pred_logits'][i].softmax(-1)
                    
                    # FIX: Explicitly grab Index 1 (Pedestrian) confidence
                    scores = probs[:, 1] 
                    # FIX: Tell metric these are Class 0
                    labels = torch.zeros_like(scores, dtype=torch.long)
                    
                    boxes = box_c_to_xy(outputs['pred_boxes'][i])
                    orig_h, orig_w = targets[i]['orig_size']
                    boxes[:, 0::2] *= orig_w
                    boxes[:, 1::2] *= orig_h

                    # FIX: No Threshold - Pass everything to metric
                    pred_list.append(dict(boxes=boxes, scores=scores, labels=labels))
                    
                    tgt_boxes = box_c_to_xy(targets[i]['boxes'])
                    tgt_boxes[:, 0::2] *= orig_w
                    tgt_boxes[:, 1::2] *= orig_h
                    
                    # FIX: Align labels to 0
                    tgt_labels = (targets[i]['labels'] - 1).to(device)
                    target_list.append(dict(boxes=tgt_boxes, labels=tgt_labels))

                metric.update(pred_list, target_list)
        
        val_map = metric.compute()
        map_50 = val_map['map_50'].item()
        map_history.append(map_50)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Val mAP@0.5: {map_50:.4f}")
    else:
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")


# %%
# Save the final model weights
torch.save(model.state_dict(), 'tinydetr_resnet18_final.pth')
print("Model saved to tinydetr_resnet18_final.pth")

# %% [markdown]
#  ## 6. Graphs & Visualization

# %%
# Plotting
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(loss_history, label='Train Loss')
ax[0].set_title('Training Loss')
ax[0].legend()
ax[0].grid(True)

epochs_x = [(i+1)*5 for i in range(len(map_history))]
ax[1].plot(epochs_x, map_history, label='Validation mAP@0.5', color='green', marker='o')
ax[1].set_title('Validation mAP@0.5')
ax[1].axhline(y=0.5, color='r', linestyle='--', label='Target (0.5)')
ax[1].legend()
ax[1].grid(True)
plt.show()

# NMS Visualization
def plot_prediction_nms(model, dataset, index, threshold=0.3):
    model.eval() 
    img_tensor, target = dataset[index]
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0).to(device))

    probas = output['pred_logits'].softmax(-1)[0, :, 1] 
    boxes = output['pred_boxes'][0]
    
    image_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    h, w, _ = image_np.shape
    
    boxes_xyxy = box_c_to_xy(boxes)
    boxes_xyxy[:, 0::2] *= w
    boxes_xyxy[:, 1::2] *= h
    
    keep = ops.nms(boxes_xyxy, probas, 0.3)
    keep = keep[probas[keep] > threshold]

    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    ax = plt.gca()

    for idx in keep:
        xmin, ymin, xmax, ymax = boxes_xyxy[idx].tolist()
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, f"{probas[idx]:.2f}", color='white', fontsize=12, fontweight='bold', bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.show()

print("Checking Final Prediction...")
plot_prediction_nms(model, test_dataset, index=0, threshold=0.7)


