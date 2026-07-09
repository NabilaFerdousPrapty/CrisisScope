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


def overlay_saliency(pil_image, grid, cmap=None, alpha=0.8, min_alpha=0.45, gamma=0.7):
    """
    Blend a (small, coarse) saliency grid over the original image as a
    Grad-CAM style heatmap.

    Default colormap is "turbo" -- a modern, perceptually-improved rainbow
    colormap (Google AI, 2019) that has largely replaced "jet" as the
    standard for heatmaps/Grad-CAM in recent papers. It keeps jet's
    intuitive blue(cold)->red(hot) reading but avoids jet's washed-out,
    perceptually flat green/yellow midsection and hard banding -- so the
    same underlying saliency values look more vivid and easier to read at a
    glance. Pass cmap=plt.get_cmap("jet") explicitly if you specifically
    want the classic look instead.

    - alpha / min_alpha: per-pixel alpha floor + ceiling so every pixel gets
      *some* color, ramping up toward `alpha` at the hottest point -- full
      coverage instead of leaving cold regions untinted.
    - gamma < 1.0 pushes mid-range saliency values toward the vivid ends of
      the colormap via `heatmap ** gamma`. Purely a display curve.
    - A light Gaussian blur is applied AFTER upsampling the coarse grid to
      full image resolution, so the heatmap reads as a smooth blob instead
      of visible blocky patch edges.
    """
    import numpy as np
    from PIL import Image, ImageFilter
    import matplotlib.pyplot as plt

    image = pil_image.convert("RGB")
    image_np = np.array(image).astype(np.float32) / 255.0

    grid = np.array(grid, dtype=np.float32)

    if grid.max() > grid.min():
        grid = (grid - grid.min()) / (grid.max() - grid.min())
    else:
        grid = np.zeros_like(grid)

    heatmap_img = Image.fromarray(np.uint8(grid * 255)).resize(
        image.size,
        resample=Image.BICUBIC,
    )
    blur_radius = max(image.size) / max(grid.shape) * 0.5
    heatmap_img = heatmap_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    heatmap = np.array(heatmap_img).astype(np.float32) / 255.0   # (H, W) in [0, 1]
    heatmap = np.clip(heatmap, 0.0, 1.0)

    heatmap_display = heatmap ** gamma

    if cmap is None:
        try:
            cmap = plt.get_cmap("turbo")
        except ValueError:
            # older matplotlib versions (<3.3) don't ship "turbo" -- fall
            # back to jet, which is always available.
            cmap = plt.get_cmap("jet")

    heatmap_rgb = cmap(heatmap_display)[..., :3]              # (H, W, 3)

    alpha_map = (min_alpha + (alpha - min_alpha) * heatmap)[..., None]  # (H, W, 1)

    overlay = (1 - alpha_map) * image_np + alpha_map * heatmap_rgb
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

    Both the vision and text saliency hooks target the SECOND-TO-LAST encoder
    layer in their respective towers (not the raw embedding table, and not
    the very last layer). Two different failure modes motivate this:

    - Hooking an EMBEDDING TABLE (vision patch-embed or text token-embed)
      means captured activations are identical for a given input regardless
      of which classifier head is attached (pure embedding lookup, upstream
      of all task-specific computation) -- attributions look nearly
      identical across the informativeness / humanitarian / damage models.

    - Hooking the LAST encoder layer's OUTPUT causes a different problem:
      CLIP's pooling reads only ONE position from that output -- the CLS
      token for vision, the EOT token for text. Every other position in
      that output tensor has no computational path to the loss, so it gets
      exactly zero gradient. Once that one pooled position is dropped (CLS
      for vision saliency, EOT/special tokens for text saliency), every
      remaining position shows a flat, all-zero attribution.

    The second-to-last layer avoids both issues: it's already been shaped by
    several attention layers (so it's task-relevant, unlike an embedding
    table), and the FINAL layer's self-attention still mixes every other
    position's representation into the pooled token's (CLS/EOT) output
    during the forward pass -- so real, non-zero gradient flows back into
    every patch/token here, unlike hooking the last layer's output directly.

    IMPORTANT: backprop is taken from the RAW LOGIT for the target class,
    NOT from the post-softmax probability. This is the standard Grad-CAM
    recipe (see the original Grad-CAM paper) and it matters a lot in
    practice: as softmax probability for the winning class approaches 1.0
    (a very confident, saturated prediction -- common in this disaster
    classification setting, e.g. 0.99+ confidence on an obviously "Severe"
    image), the GRADIENT of that probability w.r.t. earlier layers vanishes
    toward zero, because softmax's derivative flattens out near its
    extremes. Backprop through `probs[0, cls]` on a confident prediction can
    therefore produce a saliency map that's silently all-zero (no error,
    just a flat grid -> a blank/invisible heatmap once rendered), even
    though the model's underlying logits encode a perfectly good, non-flat
    pattern. Backprop through the raw `logits[0, cls]` instead sidesteps
    this entirely, since logits aren't bounded/saturated the way a softmax
    output is.
    """
    clip_model = get_clip_submodule(model)
    unfreeze_params, saved_flags = _enable_explain_grad(clip_model)

    ids = input_ids.unsqueeze(0).to(device)
    mask = attention_mask.unsqueeze(0).to(device)
    pix = pixel_values.unsqueeze(0).to(device).clone().requires_grad_(True)

    captured = {}

    # Pre-hook on the LAST vision layer == hooking the SECOND-TO-LAST layer's
    # output (its output is exactly this layer's input). Capturing here
    # (rather than layers[-1]'s output) is what keeps this from going
    # all-blue: these patch tokens still feed into layers[-1]'s
    # self-attention, which the CLS token attends to before being pooled,
    # so they carry a real, non-zero gradient.
    def vision_hook(module, inputs):
        hidden = inputs[0] if isinstance(inputs, tuple) else inputs
        hidden.retain_grad()
        captured["vision_hidden"] = hidden

    def text_hook(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        hidden.retain_grad()
        captured["text_hidden"] = hidden

    h1 = clip_model.vision_model.encoder.layers[-1].register_forward_pre_hook(vision_hook)
    h2 = clip_model.text_model.encoder.layers[-2].register_forward_hook(text_hook)

    was_training = model.training
    try:
        model.zero_grad()
        model.eval()
        logits = model(ids, mask, pix)
        probs = F.softmax(logits, dim=-1)
        pred_class = int(probs.argmax(-1).item())
        cls = target_class if target_class is not None else pred_class

        # FIX: backprop from the raw logit, not the softmax probability.
        # See the docstring note above -- this is what was silently
        # producing an all-zero grid (and therefore an invisible/blank
        # Grad-CAM overlay) on high-confidence predictions.
        score = logits[0, cls]
        score.backward()
    finally:
        h1.remove()
        h2.remove()
        _restore_explain_grad(unfreeze_params, saved_flags)
        if was_training:
            model.train()

    # --- vision saliency (Grad-CAM style, patch tokens from second-to-last layer) ---
    v_hidden = captured["vision_hidden"]
    v_grad = v_hidden.grad
    if v_grad is None:
        raise RuntimeError(
            "vision_hidden.grad is None -- the pre-hook isn't capturing a tensor "
            "that's actually used downstream. Double-check register_forward_pre_hook "
            "is attached to clip_model.vision_model.encoder.layers[-1] and that "
            "`pix` has requires_grad_(True)."
        )
    weights = (v_grad * v_hidden).sum(-1).squeeze(0)[1:]  # drop CLS token
    weights = F.relu(weights)
    n = int(weights.numel() ** 0.5)
    grid = weights[: n * n].reshape(n, n).detach().cpu().numpy()

    if grid.std() < 1e-8:
        print(f"[warn] vision grid is flat (std={grid.std():.2e}) -- Grad-CAM will "
              "render as a uniform color regardless of colormap. If this still "
              "fires after the logit fix, the prediction may be so saturated "
              "that even the raw logit's gradient underflows -- check raw "
              "logits[0, cls].item() to see how large/small the margin is.")
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)

    # --- text saliency (gradient-norm per token, from the second-to-last encoder layer) ---
    t_hidden = captured["text_hidden"]
    t_grad = t_hidden.grad
    importance = t_grad.norm(dim=-1).squeeze(0).detach().cpu().numpy()

    # Per-dimension signed Gradient x Input decomposition, taken at the
    # second-to-last text encoder layer so it reflects post-attention,
    # task-relevant representations while still receiving nonzero gradient
    # at every real token position (see note above on why the last layer
    # doesn't work). Summing this over the embedding axis for a given token recovers a signed version of
    # that token's contribution (unlike `importance` above, which is
    # magnitude-only via the gradient norm). Each dimension's value becomes
    # one "point" in a SHAP-beeswarm-style plot for that token: x = signed
    # contribution, color = that dimension's raw activation (high/low), one
    # dot per embedding dimension standing in for "one sample" in the swarm.
    contrib_per_dim = (t_grad * t_hidden).squeeze(0).detach().cpu().numpy()  # (seq_len, hidden_dim)
    activation_per_dim = t_hidden.squeeze(0).detach().cpu().numpy()  # (seq_len, hidden_dim)

    # Drop padding (via attention_mask) AND special tokens (BOS/EOS/etc, via
    # the tokenizer's special-token IDs) — attention_mask alone only removes
    # padding, since CLIP marks BOS/EOS with attention_mask=1.
    attn_mask_np = mask.squeeze(0).cpu().numpy().astype(bool)
    token_ids = ids.squeeze(0).cpu().tolist()
    special_ids = set(tokenizer.all_special_ids)
    keep_mask = np.array(
        [is_attended and (tid not in special_ids) for tid, is_attended in zip(token_ids, attn_mask_np)]
    )

    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = [t for t, k in zip(tokens, keep_mask) if k]
    importance = importance[keep_mask]
    contrib_per_dim = contrib_per_dim[keep_mask]
    activation_per_dim = activation_per_dim[keep_mask]

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
        "contrib_per_dim": contrib_per_dim,       # (num_tokens, hidden_dim) signed, for beeswarm x
        "activation_per_dim": activation_per_dim,  # (num_tokens, hidden_dim) for beeswarm color
    }