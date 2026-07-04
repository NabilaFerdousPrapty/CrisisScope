"""
Explainability utilities — a generalized version of the compute_explanations /
overlay_saliency functions from transfer_learning_methods_xai.ipynb, made to
work on a single live (image, text) sample for any of the three model
wrappers (they all end up exposing a `.clip` CLIPModel, whether raw, LoRA, or
LoRA-wrapping-a-classifier — PEFT proxies attribute access down to it).
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm


def get_clip_submodule(model):
    """Unwrap PEFT/LoRA wrapping (if any) to reach the underlying CLIPModel."""
    m = model
    for _ in range(5):
        if hasattr(m, "clip"):
            return m.clip
        if hasattr(m, "base_model"):
            m = m.base_model
        elif hasattr(m, "model"):
            m = m.model
        else:
            break
    raise AttributeError("Could not locate a .clip submodule on this model.")


def _enable_explain_grad(clip_model):
    params = [
        clip_model.vision_model.embeddings.patch_embedding.weight,
        clip_model.vision_model.embeddings.class_embedding,
        clip_model.text_model.embeddings.token_embedding.weight,
    ]
    saved = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(True)
    return params, saved


def _restore_explain_grad(params, saved):
    for p, flag in zip(params, saved):
        p.requires_grad_(flag)


def overlay_saliency(pil_image, grid, cmap=None, alpha=0.45):
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    image = pil_image.convert("RGB")
    image_np = np.array(image).astype(np.float32) / 255.0

    grid = np.array(grid, dtype=np.float32)

    if grid.max() > grid.min():
        grid = (grid - grid.min()) / (grid.max() - grid.min())
    else:
        grid = np.zeros_like(grid)

    heatmap = Image.fromarray(np.uint8(grid * 255)).resize(
        image.size,
        resample=Image.BILINEAR
    )
    heatmap = np.array(heatmap).astype(np.float32) / 255.0

    if cmap is None:
        cmap = plt.get_cmap("jet")

    heatmap_rgb = cmap(heatmap)[..., :3]

    overlay = (1 - alpha) * image_np + alpha * heatmap_rgb
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)

def compute_explanations(model, tokenizer, input_ids, attention_mask, pixel_values, device, target_class=None):
    """
    Runs one forward+backward pass and returns:
        grid            - HxW (patch-grid) normalized saliency map for the image
        tokens          - list of text tokens (special/pad tokens dropped)
        importance      - normalized per-token saliency, same length as tokens
        pred_class      - predicted class index
        probs           - full class-probability vector (numpy)
    Works for MultimodalDisasterClassifier, HumanitarianVLM(mode="lora"), and
    the LoRA-wrapped DamageSeverityClassifier.
    """
    clip_model = get_clip_submodule(model)
    unfreeze_params, saved_flags = _enable_explain_grad(clip_model)

    ids = input_ids.unsqueeze(0).to(device)
    mask = attention_mask.unsqueeze(0).to(device)
    pix = pixel_values.unsqueeze(0).to(device).clone().requires_grad_(True)

    captured = {}

    def vision_hook(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        hidden.retain_grad()
        captured["vision_hidden"] = hidden

    def text_hook(module, inp, out):
        out.retain_grad()
        captured["text_emb"] = out

    h1 = clip_model.vision_model.encoder.layers[-1].register_forward_hook(vision_hook)
    h2 = clip_model.text_model.embeddings.token_embedding.register_forward_hook(text_hook)

    was_training = model.training
    try:
        model.zero_grad()
        model.eval()
        logits = model(ids, mask, pix)
        probs = F.softmax(logits, dim=-1)
        pred_class = int(probs.argmax(-1).item())
        cls = target_class if target_class is not None else pred_class
        score = probs[0, cls]
        score.backward()
    finally:
        h1.remove()
        h2.remove()
        _restore_explain_grad(unfreeze_params, saved_flags)
        if was_training:
            model.train()

    # --- vision saliency (Grad-CAM style on last ViT layer's patch tokens) ---
    v_hidden = captured["vision_hidden"]
    v_grad = v_hidden.grad
    weights = (v_grad * v_hidden).sum(-1).squeeze(0)[1:]  # drop CLS token
    weights = F.relu(weights)
    n = int(weights.numel() ** 0.5)
    grid = weights[: n * n].reshape(n, n).detach().cpu().numpy()
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)

    # --- text saliency (gradient-norm per token embedding) ---
    t_emb = captured["text_emb"]
    t_grad = t_emb.grad
    importance = t_grad.norm(dim=-1).squeeze(0).detach().cpu().numpy()

    # Per-dimension signed Gradient x Input decomposition. Summing this over
    # the embedding axis for a given token recovers a signed version of that
    # token's contribution (unlike `importance` above, which is magnitude-only
    # via the gradient norm). Each dimension's value becomes one "point" in a
    # SHAP-beeswarm-style plot for that token: x = signed contribution,
    # color = that dimension's raw activation (high/low), one dot per
    # embedding dimension standing in for "one sample" in the swarm.
    contrib_per_dim = (t_grad * t_emb).squeeze(0).detach().cpu().numpy()  # (seq_len, embed_dim)
    activation_per_dim = t_emb.squeeze(0).detach().cpu().numpy()  # (seq_len, embed_dim)

    tok_mask = mask.squeeze(0).cpu().numpy().astype(bool)
    token_ids = ids.squeeze(0).cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = [t for t, m in zip(tokens, tok_mask) if m]
    importance = importance[tok_mask]
    contrib_per_dim = contrib_per_dim[tok_mask]
    activation_per_dim = activation_per_dim[tok_mask]

    imp_range = importance.max() - importance.min()
    if imp_range < 1e-8:
        # all tokens got ~equal gradient — avoid 0/0 -> NaN bars that render invisibly
        importance = np.zeros_like(importance)
    else:
        importance = (importance - importance.min()) / imp_range

    return {
        "grid": grid,
        "tokens": tokens,
        "importance": importance,
        "pred_class": pred_class,
        "probs": probs.detach().cpu().numpy().squeeze(0),
        "contrib_per_dim": contrib_per_dim,       # (num_tokens, embed_dim) signed, for beeswarm x
        "activation_per_dim": activation_per_dim,  # (num_tokens, embed_dim) for beeswarm color
    }