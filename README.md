# Crisis-CLIP Streamlit App

A single web app that runs your three Crisis-CLIP heads on one social-media
style (image, text) input, in sequence:

1. **Informativeness** (binary) — `MultimodalDisasterClassifier`
2. **Humanitarian category** (5-class, LoRA) — `HumanitarianVLM`
3. **Damage severity** (3-class, LoRA) — `DamageSeverityClassifier` + LoRA

Each stage shows live **Grad-CAM** (which image regions drove the prediction)
and **text-token saliency** (which words mattered), computed with a single
gradient backward pass per model — fast enough to feel real-time on GPU, and
still only ~1-3s per model on CPU.

## 1. Setup

```bash
pip install -r requirements.txt
```

(First run will download `openai/clip-vit-base-patch32` from Hugging Face —
make sure you have internet access the first time.)

## 2. checkpoints

```
checkpoints/informativeness.pth      # MultimodalDisasterClassifier state_dict
checkpoints/humanitarian_lora.pth    # HumanitarianVLM(mode="lora") state_dict
checkpoints/damage_lora.pth          # get_peft_model(DamageSeverityClassifier(), lora_cfg) state_dict
```

## 3. Run

```bash
streamlit run app.py
```

Upload an image, type a caption, click **Analyze**.

## Notes / things to double-check

- **Informativeness label mapping**: your notebook has two different label
  encodings across cells (one for the CLIP model, one for a separate
  RandomForest baseline on a different dataframe). This app uses the mapping
  that was actually applied to the data the CLIP model trained on
  (`informative → 1`, `not_informative → 0`, from cell 3 of
  `InformativenessClassify.ipynb`). If predictions come out visibly inverted
  on obvious test cases, flip `INFORMATIVE_LABELS` in `models.py`.
- **Gating behavior**: by default, if a post is predicted "Not Informative",
  the app skips the humanitarian/damage stages (matching how CrisisMMD's task
  structure is meant to be used downstream). Untick the sidebar override to
  force all three models to run regardless.
- **LoRA configs must match training exactly.** `models.py` hard-codes the
  `LoraConfig` values read out of your notebooks (`r=128, lora_alpha=256,
target_modules=["q_proj","v_proj"]`, plus `modules_to_save=["fusion",
"classifier"]` for the damage model). If you retrain with different LoRA
  hyperparameters, update `HUMANITARIAN_LORA_CONFIG` / `DAMAGE_LORA_CONFIG`
  to match, or `load_state_dict` will fail.
- Architecture + save/load logic (including the PEFT `modules_to_save`
  wrapping) was verified with a structural round-trip test before shipping,
  but the actual `.pth` files couldn't be tested end-to-end in this
  environment since it doesn't have GPU/HF-hub access — test locally with
  your real checkpoints once you run it.

## File overview

- `app.py` — Streamlit UI + pipeline orchestration
- `models.py` — the exact model classes + checkpoint loaders
- `xai.py` — Grad-CAM + text-token saliency (generalized from your

- `requirements.txt`
- `style.py` - Contains styles of the web app

## Result link
