"""
Model architectures for the Crisis-CLIP Streamlit app.

These classes are copied to match, field-for-field, the modules that were
trained and checkpointed in your three notebooks:

  - InformativenessClassify.ipynb        -> MultimodalDisasterClassifier
  - humaterian.ipynb                     -> HumanitarianVLM (mode="lora")
  - transfer_learning_methods_xai.ipynb  -> DamageSeverityClassifier (+ LoRA)

Getting the class definitions exactly right matters because torch.load()
restores a state_dict by parameter *name* — if a module is structured even
slightly differently than during training, loading will fail or silently
load into the wrong tensors.
"""

import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from peft import LoraConfig, get_peft_model

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
MAX_TEXT_LEN = 77


def _clip_features(output):
    """
    transformers <5.0: CLIPModel.get_text_features / get_image_features
    return a plain Tensor.
    transformers >=5.0: they return a BaseModelOutputWithPooling whose
    `.pooler_output` holds that same (already-projected) feature tensor.
    This makes every model in this file work with either version.
    """
    if torch.is_tensor(output):
        return output
    return output.pooler_output

# ---------------------------------------------------------------------------
# Label maps (must match training exactly)
# ---------------------------------------------------------------------------

# NOTE on informativeness labels: your notebook contains two different label
# encodings in different cells. The encoding actually used to build the
# labels the CLIP model (`model` / MultimodalDisasterClassifier) was trained
# on is in cell 3: informative -> 1, not_informative -> 0. A separate,
# later cell re-encodes a different dataframe (used only for a RandomForest
# baseline) the opposite way — that one is NOT what your checkpoint learned.
# If predictions look inverted once you test on real examples, flip
# INFORMATIVE_LABELS below and re-check.
INFORMATIVE_LABELS = {0: "Not Informative", 1: "Informative"}

HUMANITARIAN_LABELS = {
    0: "Infrastructure and Utility Damage",
    1: "Other Relevant Information",
    2: "Not Humanitarian",
    3: "Rescue, Volunteering, or Donation Effort",
    4: "Affected Individuals",
}

DAMAGE_LABELS = {0: "Severe", 1: "Mild", 2: "None"}


# ---------------------------------------------------------------------------
# 1) Informativeness (binary) — full fine-tune, last-2-layers unfrozen
# ---------------------------------------------------------------------------

class MultimodalDisasterClassifier(nn.Module):
    def __init__(self, model_id=CLIP_MODEL_ID, num_classes=2):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_id, use_safetensors=True)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        text_out = _clip_features(self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask))
        img_out = _clip_features(self.clip.get_image_features(pixel_values=pixel_values))
        text_out = text_out / text_out.norm(dim=-1, keepdim=True)
        img_out = img_out / img_out.norm(dim=-1, keepdim=True)
        combined = torch.cat((img_out, text_out), dim=1)
        return self.classifier(combined.float())


# ---------------------------------------------------------------------------
# 1b) Informativeness (binary) — LoRA variant
# ---------------------------------------------------------------------------
# This matches informativeness_lora_train.py / LoRACLIPClassifier exactly:
# r=8, alpha=16, dropout=0.1, target_modules=[q_proj, v_proj, k_proj, out_proj].
# Use this loader (not MultimodalDisasterClassifier) if your checkpoint was
# produced by that script — you can tell because the state_dict keys contain
# "lora_A" / "lora_B" / "base_model.model" instead of plain "clip.text_model...".

INFORMATIVENESS_LORA_CONFIG = dict(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
)


class InformativenessLoRAClassifier(nn.Module):
    def __init__(self, model_id=CLIP_MODEL_ID, num_classes=2):
        super().__init__()
        base_clip = CLIPModel.from_pretrained(model_id, use_safetensors=True)
        lora_cfg = LoraConfig(**INFORMATIVENESS_LORA_CONFIG)
        self.clip = get_peft_model(base_clip, lora_cfg)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        text_out = _clip_features(self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask))
        img_out = _clip_features(self.clip.get_image_features(pixel_values=pixel_values))
        text_out = F.normalize(text_out, dim=-1)
        img_out = F.normalize(img_out, dim=-1)
        combined = torch.cat((img_out, text_out), dim=1)
        return self.classifier(combined.float())


# ---------------------------------------------------------------------------
# 2) Humanitarian category (5-class) — HumanitarianVLM, mode="lora"
# ---------------------------------------------------------------------------

class GatedHumanitarianFusion(nn.Module):
    """Learns how much to trust text vs. image features dynamically."""
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.text_proj = nn.Linear(embed_dim, 512)
        self.image_proj = nn.Linear(embed_dim, 512)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=1),
        )
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, t_f, i_f):
        t_p = F.gelu(self.text_proj(t_f))
        i_p = F.gelu(self.image_proj(i_f))
        weights = self.gate(torch.cat([t_f, i_f], dim=1))
        return self.layer_norm(weights[:, 0:1] * t_p + weights[:, 1:2] * i_p)


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int = 512, num_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.net(x)


HUMANITARIAN_LORA_CONFIG = dict(
    r=128,
    lora_alpha=256,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)


class HumanitarianVLM(nn.Module):
    """mode is fixed to 'lora' here since that's the checkpoint you're using."""

    def __init__(self, num_classes: int = 5, embed_dim: int = 512, mode: str = "lora"):
        super().__init__()
        self.mode = mode
        raw_clip = CLIPModel.from_pretrained(CLIP_MODEL_ID, use_safetensors=True)

        if mode == "lora":
            lora_cfg = LoraConfig(**HUMANITARIAN_LORA_CONFIG)
            self.clip = get_peft_model(raw_clip, lora_cfg)
        else:
            raise ValueError("This app build only supports mode='lora'.")

        self.fusion = GatedHumanitarianFusion(embed_dim)
        self.classifier = ClassifierHead(embed_dim, num_classes)

    def forward(self, ids, mask, pix):
        t_f = F.normalize(_clip_features(self.clip.get_text_features(input_ids=ids, attention_mask=mask)), p=2, dim=-1)
        i_f = F.normalize(_clip_features(self.clip.get_image_features(pixel_values=pix)), p=2, dim=-1)
        fused = self.fusion(t_f, i_f)
        return self.classifier(fused)


# ---------------------------------------------------------------------------
# 3) Damage severity (3-class) — DamageSeverityClassifier + LoRA
# ---------------------------------------------------------------------------

class SymmetricFusion(nn.Module):
    """Mutual verification: Text queries Image and Image queries Text."""
    def __init__(self, embed_dim=512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 512), nn.GELU(),
            nn.Linear(512, embed_dim * 2), nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(embed_dim * 2)
        self.gamma = nn.Parameter(1e-4 * torch.ones(embed_dim * 2))

    def forward(self, t_f, i_f):
        a_t = F.scaled_dot_product_attention(t_f.unsqueeze(1), i_f.unsqueeze(1), i_f.unsqueeze(1)).squeeze(1)
        a_i = F.scaled_dot_product_attention(i_f.unsqueeze(1), t_f.unsqueeze(1), t_f.unsqueeze(1)).squeeze(1)
        combined = torch.cat((a_t, a_i), dim=1)
        return self.norm(self.gamma * (combined * self.gate(combined)) + torch.cat((t_f, i_f), dim=1))


class DamageSeverityClassifier(nn.Module):
    def __init__(self, model_id=CLIP_MODEL_ID, num_classes=3):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_id, use_safetensors=True)
        self.fusion = SymmetricFusion()
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(),
            nn.Dropout(0.5), nn.Linear(512, 128), nn.GELU(), nn.Linear(128, num_classes),
        )

    def forward(self, ids, mask, pix):
        t_f = _clip_features(self.clip.get_text_features(input_ids=ids, attention_mask=mask))
        i_f = _clip_features(self.clip.get_image_features(pixel_values=pix))
        t_f, i_f = F.normalize(t_f, p=2, dim=-1), F.normalize(i_f, p=2, dim=-1)
        return self.classifier(self.fusion(t_f, i_f))


DAMAGE_LORA_CONFIG = dict(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    # NOTE: no modules_to_save here. Your actual checkpoint's fusion/classifier
    # keys are plain ("base_model.model.fusion.gamma"), not wrapped in
    # ".original_module" / ".modules_to_save.default." — so the training
    # script did not pass modules_to_save. If you retrain, either keep it
    # this way (and manually set requires_grad=True on fusion/classifier,
    # as below) or add modules_to_save back here AND update this comment.
)


def build_damage_lora_model():
    base = DamageSeverityClassifier()
    lora_cfg = LoraConfig(**DAMAGE_LORA_CONFIG)
    model = get_peft_model(base, lora_cfg)
    # fusion/classifier aren't LoRA target_modules and aren't in
    # modules_to_save, so PEFT's "freeze everything except LoRA" default
    # would leave them frozen. Un-freeze them to match how they were
    # actually trained (this only matters if you resume training; for
    # inference-only loading it's harmless either way).
    for p in model.base_model.model.fusion.parameters():
        p.requires_grad = True
    for p in model.base_model.model.classifier.parameters():
        p.requires_grad = True
    return model


# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------


def _read_checkpoint_state(ckpt_path):
    """Load a checkpoint on CPU with lower peak memory where supported."""
    load_kwargs = {
        "map_location": "cpu",
        "weights_only": True,
    }
    try:
        state = torch.load(ckpt_path, mmap=True, **load_kwargs)
    except (TypeError, RuntimeError, ValueError):
        state = torch.load(ckpt_path, **load_kwargs)

    # Support both a raw state_dict and common training-checkpoint wrappers.
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = state.get(key)
            if isinstance(candidate, dict):
                state = candidate
                break

    if not isinstance(state, dict):
        raise TypeError(
            f"Checkpoint {ckpt_path!s} does not contain a PyTorch state_dict."
        )

    # DataParallel/DistributedDataParallel checkpoints often add "module.".
    if state and all(isinstance(k, str) and k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}

    return state


def _load_state_and_finalize(model, state, device):
    """Load weights, move to the selected device, and release checkpoint RAM."""
    try:
        model.load_state_dict(state, strict=True, assign=True)
    except TypeError:
        # `assign` is unavailable in older PyTorch releases.
        model.load_state_dict(state, strict=True)

    del state
    gc.collect()

    model = model.to(device)
    model.eval()
    return model


def load_informativeness_model(ckpt_path, device):
    """
    Auto-detect which informativeness architecture produced the checkpoint.
    LoRA checkpoints contain lora_A/lora_B keys; full fine-tune checkpoints do
    not. The checkpoint is memory-mapped on CPU when PyTorch supports it.
    """
    state = _read_checkpoint_state(ckpt_path)
    is_lora = any("lora_A" in key or "lora_B" in key for key in state)

    if is_lora:
        model = InformativenessLoRAClassifier()
    else:
        model = MultimodalDisasterClassifier()

    return _load_state_and_finalize(model, state, device)


def load_humanitarian_model(ckpt_path, device):
    state = _read_checkpoint_state(ckpt_path)
    model = HumanitarianVLM(mode="lora")
    return _load_state_and_finalize(model, state, device)


def load_damage_model(ckpt_path, device):
    state = _read_checkpoint_state(ckpt_path)
    model = build_damage_lora_model()
    return _load_state_and_finalize(model, state, device)
