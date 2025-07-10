from sklearn.model_selection import train_test_split
from split_dataset import train_speaker, test_speaker, asv_balanced
from transformers import Wav2Vec2Model
import torch.nn.functional as F
from dataAudio import AudioConfig, AudioProcessor, DeepfakeDataset
import argparse
from ASVSpoofDataset import ASVspoofDataset, ASVSpoofProcessor
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, precision_recall_fscore_support

# ===================================================================
#  Configuration & Logging (Assuming these are defined elsewhere)
# ===================================================================
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")


def temporal_sparsity_loss(attn_weights: torch.Tensor) -> torch.Tensor:
    """Stable entropy-based sparsity loss"""
    # attn_weights: [B, heads, Q, K]
    probs = attn_weights.clamp(min=1e-12)  # Avoid log(0)
    entropy = - (probs * torch.log(probs)).sum(dim=-1)  # Sum over keys
    max_entropy = torch.log(torch.tensor(attn_weights.shape[-1], device=attn_weights.device))
    normalized_entropy = entropy / max_entropy
    return normalized_entropy.mean()  # Minimize this = sparser attention

def head_diversity_loss(attn_weights: torch.Tensor) -> torch.Tensor:
    """Memory-efficient pairwise diversity loss"""
    B, H, Q, K = attn_weights.shape
    diversity = 0
    for i in range(H):
        for j in range(i+1, H):
            diff = attn_weights[:, i] - attn_weights[:, j]
            diversity += diff.abs().mean()
    return diversity / (H*(H-1)/2 + 1e-8)

def content_consistency_loss(attn_weights: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Feature similarity alignment without new projections"""
    # attn_weights: [B, heads, Q, K]
    # features: [B, C, H, W] (ResNet output before flattening)
    
    # Flatten spatial dimensions
    B, C, H, W = features.shape
    queries = features.view(B, C, H*W).permute(0, 2, 1)  # [B, H*W, C]
    
    # Compute feature similarity
    sim_matrix = torch.cdist(queries, queries, p=2)  # [B, Q, Q]
    sim_matrix = 1 / (1 + sim_matrix)  # Convert distance to similarity
    
    # Compute attention similarity
    attn_sim = torch.einsum('bhqk,bhqk->bhq', attn_weights, attn_weights)  # [B, heads, Q]
    
    # Align feature and attention similarities
    return F.mse_loss(attn_sim.mean(dim=1), sim_matrix.mean(dim=-1))
# ===================================================================
#  1. Core Building Blocks for the Collaborative Network
# ===================================================================

class CrossAttention(nn.Module):
    """
    A simple but effective cross-attention module.
    Query attends to Key-Value pairs.
    """
    def __init__(self, query_dim: int, key_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, query_dim) # Project back to query's dimension
        self.scale = self.head_dim ** -0.5

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        batch_size, query_len, _ = query.shape
        _, key_len, _ = key.shape

        q = self.query_proj(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(key).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(value).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        attended_v = torch.matmul(attn_weights, v)
        attended_v = attended_v.transpose(1, 2).contiguous().view(batch_size, query_len, -1)

        return self.out_proj(attended_v), {
            'attn_weights': attn_weights,
            'queries': query,  # Original queries
            'keys': key        # Original keys
        }


class WSFM(nn.Module):
    """
    Waveform-Spectrogram Fusion Module (WSFM) - CORRECTED VERSION

    Performs UNIDIRECTIONAL cross-attention. The trainable spectrogram branch
    (student) attends to the frozen Wav2Vec branch (teacher) to gather context.
    The Wav2Vec features are passed through UNMODIFIED.
    """
    def __init__(self, wav2vec_dim: int, spec_dim: int, hidden_dim: int = 256):
        super().__init__()
        # We only need one-way attention: student queries the teacher
        self.spec_attends_to_wav2vec = CrossAttention(query_dim=spec_dim, key_dim=wav2vec_dim, hidden_dim=hidden_dim)

        # We only need to normalize the features we are updating: the spectrogram features
        self.norm = nn.LayerNorm(spec_dim)

    def forward(self, f_wav2vec: torch.Tensor, f_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            f_wav2vec (torch.Tensor): Teacher features [B, T, D_w]. These will NOT be modified.
            f_spec (torch.Tensor): Student features [B, C, H, W]. These WILL be updated.
        
        Returns:
            A tuple containing:
            - The original, unmodified f_wav2vec tensor.
            - The updated f_spec tensor.
        """
        batch_size, spec_c, spec_h, spec_w = f_spec.shape
        
        # --- Spectrogram branch queries Wav2Vec branch ---
        
        # 1. Reshape spec features into a sequence for attention
        f_spec_seq = f_spec.flatten(2).permute(0, 2, 1) # [B, H*W, C]
        
        # 2. Perform cross-attention and CORRECTLY UNPACK THE OUTPUT
        attention_output, attn_data = self.spec_attends_to_wav2vec(
            query=f_spec_seq, 
            key=f_wav2vec, 
            value=f_wav2vec
        )
        
        # Check if the output is a tuple (context, weights) and take only the context
        if isinstance(attention_output, tuple):
            spec_context = attention_output[0]
        else:
            spec_context = attention_output

        # 3. Apply residual connection and normalization
        f_spec_updated_seq = self.norm(f_spec_seq + spec_context)
        
        # 4. Reshape back to 2D feature map format
        f_spec_updated = f_spec_updated_seq.permute(0, 2, 1).view(batch_size, spec_c, spec_h, spec_w)

        # Return the UNMODIFIED teacher features and the UPDATED student features
        return f_wav2vec, f_spec_updated, attn_data


class ResidualBlock(nn.Module):
    # Your existing ResidualBlock code - no changes needed.
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out

# ===================================================================
#  2. The Main Collaborative Network
# ===================================================================

class MultiViewCollaborativeNet(nn.Module):
    def __init__(self, wav2vec_model_name: str = "facebook/wav2vec2-large-xlsr-53", num_spec_blocks: int = 24):
        super().__init__()
        attn_implementation = "sdpa"  # Default sicuro e performante per CPU e MPS

        # Sovrascrivi solo se CUDA e Flash Attention sono disponibili
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            try:
                import flash_attn
                attn_implementation = "flash_attention_2"
            except ImportError:
                pass # Rimane "sdpa"

        print(f"--- Utilizzando l'implementazione di Attention: {attn_implementation} ---")
                
        # --- Branch 1: Wav2Vec Expert (Frozen) ---
        logger.info(f"Loading frozen Wav2Vec model: {wav2vec_model_name}")
        self.wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_model_name, attn_implementation=attn_implementation)
        for param in self.wav2vec.parameters():
            param.requires_grad = False
        self.wav2vec.eval() # Put in eval mode to disable dropout, etc.
        
        wav2vec_dim = self.wav2vec.config.hidden_size # e.g., 1024 for large-xls-r
        num_wav2vec_layers = self.wav2vec.config.num_hidden_layers # Should be 24

        if num_spec_blocks != num_wav2vec_layers:
            raise ValueError(f"num_spec_blocks ({num_spec_blocks}) must match num_wav2vec_layers ({num_wav2vec_layers})")

        # --- Branch 2: Spectrogram Expert (Trainable) ---
        # We build a deep ResNet with a number of blocks matching Wav2Vec layers.
        # We need to carefully manage channels and strides.
        initial_channels = 64
        spec_channels = [64, 128, 256, 512, 1024] # Example channel progression
        blocks_per_stage = [4, 4, 6, 6, 4] # Must sum to num_spec_blocks (24)
        
        self.spec_initial_conv = nn.Sequential(
            nn.Conv2d(1, initial_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.spec_blocks = nn.ModuleList()
        self.wsfms = nn.ModuleList()
        current_channels = initial_channels
        for i, num_blocks in enumerate(blocks_per_stage):
            out_channels = spec_channels[i]
            for j in range(num_blocks):
                stride = 2 if (j == 0 and i > 0) else 1
                self.spec_blocks.append(ResidualBlock(current_channels, out_channels, stride))
                self.wsfms.append(
                WSFM(wav2vec_dim=wav2vec_dim, spec_dim=out_channels, hidden_dim=512)
            )
                current_channels = out_channels
        
        spec_dim = current_channels # Final channel dim of spec branch

        # --- Fusion Modules (Trainable) ---


        # --- Final Heads (Trainable) ---
        self.final_pool_w = nn.AdaptiveAvgPool1d(1)
        self.final_pool_s = nn.AdaptiveAvgPool2d((1, 1))
        
        self.aux_classifier_s = nn.Linear(spec_dim, 2)


    def forward(self, x_wav2vec_input: torch.Tensor, x_spec: torch.Tensor):
        # Ensure spec input is 4D
        if x_spec.dim() == 3:
            x_spec = x_spec.unsqueeze(1)

        # --- Initial Feature Extraction ---
        # Wav2Vec CNN feature extractor + positional embeddings
        with torch.no_grad():
            f_w = self.wav2vec.feature_extractor(x_wav2vec_input)
            f_w = f_w.transpose(1, 2)
            f_w, _ = self.wav2vec.feature_projection(f_w)
            if self.wav2vec.config.mask_time_prob > 0:
                 # This part is for training Wav2Vec, we can skip it for inference
                 pass
            
        f_s = self.spec_initial_conv(x_spec)

        all_temporal_losses = []  # NEW: Track individual losses
        all_diversity_losses = []  # NEW
        all_consistency_losses = []  # NEW

        # --- Core Collaborative Loop ---
        for i in range(len(self.spec_blocks)):
            # Process one block from each branch
            # Corrected Unidirectional Flow
            with torch.no_grad():
                # Get the expert's output for this layer, but don't modify it further
                w_out = self.wav2vec.encoder.layers[i](f_w)[0]



            # Process the student branch
            s_out = self.spec_blocks[i](f_s)

            # The WSFM uses the expert's output to guide the student, but only updates the student's features
            w_out, s_out, attn_data = self.wsfms[i](w_out, s_out) # Assuming WSFM outputs a new f_s

            attn_weights=attn_data['attn_weights']
            temporal_loss = temporal_sparsity_loss(attn_weights)
            diversity_loss = head_diversity_loss(attn_weights)
            consistency_loss = content_consistency_loss(attn_weights, s_out)  # s_out is ResNet feature
            

            
            # Store individual losses
            all_temporal_losses.append(temporal_loss)
            all_diversity_losses.append(diversity_loss)
            all_consistency_losses.append(consistency_loss)

            # Update features for the next iteration
            f_w = w_out  # f_w remains the pure, unmodified output from the expert
            f_s = s_out  # f_s is the collaboratively refined feature map

        # --- Final Pooling and Classification ---
        f_w_vec = self.final_pool_w(f_w.permute(0, 2, 1)).squeeze(-1)
        f_s_vec = self.final_pool_s(f_s).flatten(1)


        logits_s = self.aux_classifier_s(f_s_vec)

        return {
            "final_logits": logits_s,
            "feat_w": f_w_vec,
            "feat_s": f_s_vec,
            "temporal_losses": torch.stack(all_temporal_losses),  # NEW
            "diversity_losses": torch.stack(all_diversity_losses),  # NEW
            "consistency_losses": torch.stack(all_consistency_losses),  # NEW
        }

# ===================================================================
#  3. The Collaborative Loss Function
# ===================================================================

class CollaborativeLoss(nn.Module):
    def __init__(self,
                 class_weights: Optional[torch.Tensor] = None,
                 w_cls: float = 1.0,      # Weight for final classification
                 w_intra_s: float = 0.6,  # Weight for student's metric learning (inner-view)
                 w_attn: float = 0.2,
                 temp: float = 0.5,
                 margin: float = 0,
                 ema_decay: float = 0.99,
                 min_diversity: float = 0.01,
                 focus_factor: float =5.0):
        super().__init__()
        if class_weights is not None:
            logger.info("CollaborativeLoss: Initializing with class weights for CrossEntropyLoss.")
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        # Pesi per bilanciare le componenti della loss
        self.w_cls = w_cls           # Peso per la classificazione principale
        self.w_intra_s = w_intra_s   # Peso per la loss metrica dello spettrogramma
        self.w_attn=w_attn
        self.temp = temp
        self.margin=margin
        self.focus_factor = focus_factor
                # Centroid stability parameters
        self.ema_decay = ema_decay
        self.min_diversity = min_diversity
        
        # Initialize EMA buffers with placeholders
        self.register_buffer('real_center_ema', None)
        self.register_buffer('fake_center_ema', None)
        self.register_buffer('diversity_ema', torch.tensor(0.5))
        self.register_buffer('initialized', torch.tensor(False))
        

        # Parametri per la loss metrica (precedentemente inner-view)



    def _artifact_contrastive_loss(self, features: torch.Tensor, 
                                 labels: torch.Tensor) -> torch.Tensor:
        real_mask = labels == 0
        fake_mask = labels == 1
        
        # Skip if insufficient samples
        n_real = real_mask.sum().item()
        n_fake = fake_mask.sum().item()
        if n_real < 2 or n_fake < 1:
            return torch.tensor(0.0, device=features.device)
        
        # ===== ROBUST CENTROID INITIALIZATION ===== #
        if not self.initialized:
            self.real_center_ema = features[real_mask].mean(dim=0, keepdim=True).detach()
            self.fake_center_ema = features[fake_mask].mean(dim=0, keepdim=True).detach()
            
            # Compute initial diversity using MAD (more robust than std)
            real_med = features[real_mask].median(dim=0).values
            real_mad = (features[real_mask] - real_med).abs().median().clamp(min=self.min_diversity)
            
            fake_med = features[fake_mask].median(dim=0).values
            fake_mad = (features[fake_mask] - fake_med).abs().median().clamp(min=self.min_diversity)
            
            self.diversity_ema = ((real_mad + fake_mad) / 2).detach()
            self.initialized = torch.tensor(True)
            return torch.ones(1, device=features.device)  # Warm-start as tensor
        
        # ===== COSINE DISTANCE WITH FOCUSED MARGIN ===== #
        # Normalize features and centroids
        features_norm = F.normalize(features, p=2, dim=1)
        real_center_norm = F.normalize(self.real_center_ema, p=2, dim=1)
        fake_center_norm = F.normalize(self.fake_center_ema, p=2, dim=1)
        
        # Compute cosine distances (1 - similarity)
        real_dists = 1 - torch.mm(features_norm, real_center_norm.t()).squeeze(1)
        fake_dists = 1 - torch.mm(features_norm, fake_center_norm.t()).squeeze(1)
        
        # Adaptive margin based on sample difficulty
        with torch.no_grad():
            # Dynamic margin scaling for hard samples
            difficulty = (real_dists - fake_dists).abs()
            margin_scale = 1 + (self.focus_factor * torch.sigmoid(difficulty * 5))
            adaptive_margin = self.margin * margin_scale
        
        # ===== FOCUSED LOSS CALCULATION ===== #
        losses = []
        for i in range(len(features)):
            if labels[i] == 0:  # Real sample
                violation = fake_dists[i] - real_dists[i] + adaptive_margin[i]
                if violation > 0:
                    # Quadratic penalty for hard samples
                    loss = violation ** 2
                else:
                    loss = torch.tensor(0.0, device=features.device)  # FIX: Use tensor
            else:  # Fake sample
                violation = real_dists[i] - fake_dists[i] + adaptive_margin[i]
                if violation > 0:
                    loss = violation ** 2
                else:
                    loss = torch.tensor(0.0, device=features.device)  # FIX: Use tensor
                    
            losses.append(loss)
        
        # Convert list to tensor stack
        loss_tensor = torch.stack(losses)  # Now all elements are tensors
        
        # Update centroids (after loss calculation)
        with torch.no_grad():
            real_center_batch = features[real_mask].mean(dim=0, keepdim=True)
            fake_center_batch = features[fake_mask].mean(dim=0, keepdim=True)
            
            # Update EMAs
            self.real_center_ema = (self.ema_decay * self.real_center_ema + 
                                   (1 - self.ema_decay) * real_center_batch)
            self.fake_center_ema = (self.ema_decay * self.fake_center_ema + 
                                   (1 - self.ema_decay) * fake_center_batch)
            
            # Update diversity using IQR (interquartile range)
            real_sorted, _ = features[real_mask].sort(dim=0)
            q1 = real_sorted[n_real//4]
            q3 = real_sorted[min(3*n_real//4, n_real-1)]
            real_iqr = (q3 - q1).mean().clamp(min=self.min_diversity)
            
            fake_sorted, _ = features[fake_mask].sort(dim=0)
            q1 = fake_sorted[n_fake//4]
            q3 = fake_sorted[min(3*n_fake//4, n_fake-1)]
            fake_iqr = (q3 - q1).mean().clamp(min=self.min_diversity)
            
            self.diversity_ema = (self.ema_decay * self.diversity_ema +
                                 (1 - self.ema_decay) * (real_iqr + fake_iqr)/2)
        
        return loss_tensor.mean()


    def forward(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        
        # 1. Main classification loss (Essential)
        loss_cls = self.cross_entropy_loss(outputs["final_logits"], labels)

        loss_intra_s = self._artifact_contrastive_loss(
            features=outputs["feat_s"], 
            labels=labels
        )
        loss_attn = 0

        temporal_losses = outputs["temporal_losses"].sum()
        diversity_loss = outputs["diversity_losses"].sum()
        consistency_loss = outputs["consistency_losses"].sum()
        loss_attn=(
                consistency_loss * 0.01 + 
                diversity_loss * 0.01 +
                temporal_losses * 0.01
            )

        # Final weighted sum
        total_loss = (self.w_cls * loss_cls +
                      self.w_intra_s * loss_intra_s +
                      self.w_attn * loss_attn)
        
        return total_loss

# ===================================================================
#  4. Updated Training & Evaluation Loops
# ===================================================================


def collate_fn_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    # Custom collate because default_collate can't handle dicts of tensors well
    elem = batch[0]
    collated_batch = {}
    for key in elem:
        if key == 'features':
            collated_batch[key] = {k: torch.stack([d[key][k] for d in batch]) for k in elem[key]}
            if collated_batch['features']['wav2vec_input'].shape != 2:
                collated_batch['features']['wav2vec_input'] = collated_batch['features']['wav2vec_input'].squeeze(1)
        else:
            collated_batch[key] = torch.utils.data.dataloader.default_collate([d[key] for d in batch])
            if collated_batch['features']['wav2vec_input'].shape != 2:
                collated_batch['features']['wav2vec_input'] = collated_batch['features']['wav2vec_input'].squeeze(1)
    return collated_batch['features'], collated_batch['labels']


def train_collaborative(
        model: MultiViewCollaborativeNet,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epochs: int = 10,
        save_path: str = "best_collaborative_model.pth"
) -> Dict[str, List[float]]:
    
    criterion = CollaborativeLoss().to(device)
    best_val_eer = 1.0 # Lower is better for EER
    metrics = {
        "train_loss": [], "val_loss": [], "val_acc": [],
        "val_precision": [], "val_recall": [], "val_eer": []
    }

    for epoch in range(epochs):
        model.train()
        # The frozen wav2vec part should remain in eval mode
        model.wav2vec.eval()
        
        epoch_train_loss = 0.0
        num_train_samples = 0

        for batch_idx, batch_data in enumerate(train_loader):
            if batch_data is None:
                continue

            features_dict, labels = batch_data
            # IMPORTANT: Your dataset must now provide both mel and wav2vec_input
            mel_spectrograms = features_dict["mel"].to(device)
            wav2vec_inputs = features_dict["wav2vec_input"].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs_dict = model(wav2vec_inputs, mel_spectrograms)
            loss = criterion(outputs_dict, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * labels.size(0)
            num_train_samples += labels.size(0)

            if batch_idx > 0 and batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch}/{epochs-1} | Batch {batch_idx}/{len(train_loader)-1} | Train Loss: {loss.item():.4f}")

        avg_epoch_train_loss = epoch_train_loss / num_train_samples if num_train_samples > 0 else 0
        metrics["train_loss"].append(avg_epoch_train_loss)

        if val_loader:
            val_results = evaluate_collaborative(model, val_loader, criterion)
            metrics["val_loss"].append(val_results["loss"])
            metrics["val_acc"].append(val_results["acc"])
            metrics["val_precision"].append(val_results["precision"])
            metrics["val_recall"].append(val_results["recall"])
            metrics["val_eer"].append(val_results["eer"])

            logger.info(
                f"Epoch {epoch}/{epochs-1} Summary:\n"
                f"  Train Loss: {avg_epoch_train_loss:.4f}\n"
                f"  Val Loss: {val_results['loss']:.4f}, Acc: {val_results['acc']:.4f}, EER: {val_results['eer']:.4f}"
            )

            if val_results["eer"] < best_val_eer:
                best_val_eer = val_results["eer"]
                torch.save(model.state_dict(), save_path)
                logger.info(f"New best model saved with validation EER: {best_val_eer:.4f}")

            if scheduler:
                scheduler.step(val_results["eer"])
    
    logger.info(f"Training finished. Best Validation EER: {best_val_eer:.4f}")
    return metrics

def train_On_ASVspoof2021(dataset, batch_size, num_workers):
    train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    config = AudioConfig()
    asv_processor = ASVSpoofProcessor(config, wav2vec_model_name="C:\\Users\dmc\PycharmProjects\CASA-FVAB\wav2vec2-xlsr")
    asv_train = ASVspoofDataset("", train_dataset, 5, asv_processor, "train")
    asv_valid = ASVspoofDataset("", valid_dataset, 5, asv_processor, "eval")

    train_loader = torch.utils.data.DataLoader(asv_train, batch_size=batch_size, shuffle=True,
                                               num_workers= num_workers, collate_fn=collate_fn_skip_none)
    valid_loader = torch.utils.data.DataLoader(asv_valid, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, collate_fn=collate_fn_skip_none)

    return train_loader, valid_loader

def evaluate_collaborative(
        model: MultiViewCollaborativeNet,
        data_loader: torch.utils.data.DataLoader,
        criterion: CollaborativeLoss,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_labels, all_scores = [], []

    with torch.no_grad():
        for batch_data in data_loader:
            if batch_data is None: continue
            
            features_dict, labels = batch_data
            mel_spectrograms = features_dict["mel"].to(device)
            wav2vec_inputs = features_dict["wav2vec_input"].to(device)
            labels = labels.to(device)

            outputs_dict = model(wav2vec_inputs, mel_spectrograms)
            loss = criterion(outputs_dict, labels)
            total_loss += loss.item() * labels.size(0)

            probs = torch.softmax(outputs_dict["final_logits"], dim=1)
            scores = probs[:, 1] # Probability of being spoof (class 1)
            
            all_labels.append(labels.cpu().numpy())
            all_scores.append(scores.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)
    
    avg_loss = total_loss / len(all_labels)
    all_preds = (all_scores > 0.5).astype(int)
    
    accuracy = (all_preds == all_labels).mean()
    precision, recall, _, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    return {
        "loss": avg_loss,
        "acc": accuracy,
        "precision": precision,
        "recall": recall,
        "eer": eer
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-View Collaborative Training for Deepfake Detection")

    # --- Data Arguments ---
    parser.add_argument("--metadata_path", type=Path, required=True, help="Path to the CSV metadata file.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Root directory of audio data.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. May need to be smaller due to model size.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--test_split_ratio", type=float, default=0.2, help="Ratio for test set split.")
    
    # --- Optimizer and Scheduler Arguments ---
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for trainable parts.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization.")

    # --- Model Architecture Arguments ---
    parser.add_argument("--wav2vec_model_name", type=str, default="facebook/wav2vec2-large-xlsr-53", help="Hugging Face model name for the frozen branch.")
    # Note: We removed ResNet-specific args as the architecture is now more fixed.
    # You could add args for hidden_dims in WSFM, etc., if needed.

    args = parser.parse_args()

    logger.info(f"Starting multi-view collaborative training with args: {args}")
    logger.info(f"Using device: {device}")

    train_loader, test_loader = None, None
    
    # Set multiprocessing start method for CUDA
    if args.num_workers > 0 and device.type == 'cuda':
        if torch.multiprocessing.get_start_method(allow_none=True) != 'spawn':
            torch.multiprocessing.set_start_method('spawn', force=True)
    print("Inserire Dataset d'addestramento: 0(In_The_Wild), 1(ASVspoof2021)")
    match = int(input())
    if match == 0:
        # --- Initialize Audio Processor with Wav2Vec ---
        audio_conf = AudioConfig() # Define your audio config parameters here
        processor = AudioProcessor(audio_conf, wav2vec_model_name=args.wav2vec_model_name)

        # --- Load and Split Metadata ---
        logger.info("--- Loading and Splitting Metadata ---")
        full_metadata_df = pd.read_csv(args.metadata_path)
        if full_metadata_df.empty:
            logger.error(f"Metadata CSV file at {args.metadata_path} is empty. Exiting.")
            return

        # Ensure 'label' column exists and is valid
        if 'label' not in full_metadata_df.columns:
            logger.error("Metadata CSV must contain a 'label' column ('bona-fide' or 'spoof'). Exiting.")
            return
        if not all(label in ['bona-fide', 'spoof'] for label in full_metadata_df['label'].unique()):
            logger.error("The 'label' column must only contain 'bona-fide' or 'spoof' values. Exiting.")
            return
        train_metadata_df = full_metadata_df[full_metadata_df['speaker'].isin(train_speaker)].reset_index(drop=True)
        test_metadata_df = full_metadata_df[full_metadata_df['speaker'].isin(test_speaker)].reset_index(drop=True)

        # --- Create Datasets and DataLoaders ---
        logger.info("--- Creating Datasets and DataLoaders ---")
        train_dataset = DeepfakeDataset(args.data_dir, train_metadata_df, processor, augment=True)
        test_dataset = DeepfakeDataset(args.data_dir, test_metadata_df, processor, augment=False) if not test_metadata_df.empty else None

        if len(train_dataset) == 0:
            logger.error("Training dataset is empty. Exiting.")
            return

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_none
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_none
        ) if test_dataset else None
    elif match == 1:
        train_loader, test_loader = train_On_ASVspoof2021(asv_balanced, args.batch_size, args.num_workers)

    # --- Initialize the Collaborative Model ---
    logger.info("--- Initializing Multi-View Collaborative Network ---")
    model = MultiViewCollaborativeNet(wav2vec_model_name=args.wav2vec_model_name).to(device)
    
    # Log model parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total model parameters: {total_params:,}")
    logger.info(f"Trainable model parameters: {trainable_params:,}")
    
    # --- Setup Optimizer and Scheduler ---
    # IMPORTANT: We only optimize the trainable parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    # Scheduler now steps based on EER (lower is better)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5)

    # --- Start Training ---
    logger.info("--- Starting Training ---")
    save_path = "best_collaborative_model2.pth"
    
    # Use the new training function
    train_collaborative(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        save_path=save_path
    )

    # --- Final Evaluation on Test Set ---
    if test_loader:
        logger.info("--- Loading best model for final evaluation on the test set ---")
        if not Path(save_path).exists():
            logger.warning(f"Best model file {save_path} not found. Cannot perform final evaluation.")
            return
            
        # Re-initialize model to load state dict
        final_model = MultiViewCollaborativeNet(wav2vec_model_name=args.wav2vec_model_name).to(device)
        final_model.load_state_dict(torch.load(save_path, map_location=device))
        logger.info(f"Successfully loaded best model from {save_path}")
        
        # We need a criterion instance for evaluation, even if we only care about metrics
        eval_criterion = CollaborativeLoss().to(device)
        
        test_results = evaluate_collaborative(final_model, test_loader, eval_criterion)
        
        logger.info(
            f"Final Test Results:\n"
            f"  Accuracy: {test_results['acc']:.4f}\n"
            f"  Loss: {test_results['loss']:.4f}\n"
            f"  Precision: {test_results['precision']:.4f}\n"

            f"  Recall: {test_results['recall']:.4f}\n"
            f"  EER: {test_results['eer']:.4f}"
        )
    else:
        logger.info("Skipping final evaluation as no test set was provided.")

if __name__ == "__main__":
    main()