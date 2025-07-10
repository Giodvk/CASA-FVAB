from huggingface_hub import snapshot_download
import os
local_path = 'C:\\Users\dmc\.cache\huggingface\hub\models--facebook--wav2vec2-large-xlsr-53'
os.makedirs(local_path, exist_ok=True)


snapshot_download(
    repo_id="facebook/wav2vec2-large-xlsr-53",
    local_dir=local_path,
    local_dir_use_symlinks=False
)
print(f"Download completato in: {local_path}")