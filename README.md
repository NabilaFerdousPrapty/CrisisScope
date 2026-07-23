# CrisisCLIP-X Streamlit deployment

Upload these files to the same GitHub repository directory:

- `app.py`
- `models.py`
- `xai.py`
- `style.py`
- `requirements.txt`
- `runtime.txt`

Do not rename `models.py`, `xai.py`, or `style.py`. `app.py` imports those exact
module names.

The app keeps the local checkpoint paths:

- `checkpoints/best_informativeness_lora.pth`
- `checkpoints/best_humanitarian_enhanced_lora.pth`
- `checkpoints/best_damage_model.pth`

If a local file is absent, Automatic mode downloads the matching file from:

`nabila-prapty/disaster-classification-models`

A public Hugging Face repository does not need a token. If the repository is
private, add the following in Streamlit Community Cloud app settings under
Secrets:

```toml
HF_TOKEN = "hf_your_token"
```

Deploy `app.py` as the entrypoint. After replacing the files, reboot the app so
Streamlit reinstalls dependencies from `requirements.txt`.
