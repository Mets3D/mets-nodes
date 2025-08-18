import os
import re
import requests
import folder_paths
from urllib.parse import urlparse, parse_qs


class DownloadCivitaiModel:
    """
    Download a .safetensors file from CivitAI, unless it is already present at the specified location.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"multiline": False, "tooltip": "Your CivitAI account API key. You can easily generate one in your account settings."}),
                "model_url": ("STRING", {"tooltip": "URL of the model's home page. If this contains a modelVersionID AND a version with that ID actually exists, then the version input field below will be ignored."}),
                "subdir": ("STRING", {"default": "loras", "tooltip": "Subdirectory relative to ComfyUI's install dir to which the model should be downloaded."}),
            },
            "optional": {
                "filename": ("STRING", {"tooltip": "(Optional.) Filename (without extension) to rename the file to."}),
                "version": ("STRING", {"tooltip": "(Optional.) Name (not ID) of the model version to download. Many Loras might come with versions like SDXL/Illustrious/Pony, or V1, V2, V3. If you enter nothing, the 'last' (left-most) will be downloaded."}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "download_model"
    CATEGORY = "MetsNodes"

    def download_model(self, model_url, subdir, api_token, filename="", version=""):
        # --- 1. Extract model ID ---
        m = re.search(r"/models/(\d+)(?:/|$|/?)", model_url)
        if not m:
            raise ValueError("Invalid CivitAI model URL")
        model_id = int(m.group(1))

        # --- 2. Check for modelVersionId in query string ---
        parsed_url = urlparse(model_url)
        qs = parse_qs(parsed_url.query)
        version_id_from_url = None
        if "modelVersionId" in qs:
            try:
                version_id_from_url = int(qs["modelVersionId"][0])
            except ValueError:
                pass

        headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}

        # --- 3. Fetch model details ---
        api_url = f"https://civitai.com/api/v1/models/{model_id}"
        resp = requests.get(api_url, headers=headers)
        if resp.status_code == 401:
            raise RuntimeError(
                "CivitAI API error: 401 Unauthorized.\n\n"
                "You must enter an API key, which you can get from your account settings on Civitai."
            )
        if resp.status_code != 200:
            raise RuntimeError(f"CivitAI API error: {resp.status_code} {resp.text}")

        model_data = resp.json()

        # --- 4. Select version ---
        versions = model_data.get("modelVersions", [])
        if not versions:
            raise RuntimeError("No versions found for model")

        selected_version = None

        # 4a. Try version ID from URL
        if version_id_from_url:
            for v in versions:
                if v.get("id") == version_id_from_url:
                    selected_version = v
                    break
            if selected_version:
                print(f"[CivitAI] Using version from URL: {selected_version.get('name', 'Unknown')}")
        # 4b. If no valid version ID from URL, use version input field
        if not selected_version and version:
            for v in versions:
                if v.get("name", "").lower() == version.lower():
                    selected_version = v
                    break
            if not selected_version:
                raise RuntimeError(f"Specified version '{version}' not found for this model")
        # 4c. Fallback to default (latest)
        if not selected_version:
            selected_version = versions[0]
            print(f"[CivitAI] No version specified, using default latest: {selected_version.get('name', 'Unknown')}")

        # --- 5. Find primary .safetensors file ---
        files = selected_version.get("files", [])
        primary_file = None
        for f in files:
            if f.get("primary", False) and f["name"].endswith(".safetensors"):
                primary_file = f
                break
        if not primary_file:
            raise RuntimeError("No primary .safetensors file found for this version")

        download_url = primary_file["downloadUrl"]
        orig_name = primary_file["name"]
        _, ext = os.path.splitext(orig_name)

        # --- 6. Build target path ---
        models_base_dir = os.path.dirname(folder_paths.get_folder_paths("checkpoints")[0])
        target_dir = os.path.join(models_base_dir, subdir)
        os.makedirs(target_dir, exist_ok=True)

        target_name = filename + ext if filename else orig_name
        target_path = os.path.join(target_dir, target_name)

        # --- 7. Download if missing ---
        if os.path.exists(target_path):
            print(f"[CivitAI] File already exists: {target_path}")
            return ()

        print(f"[CivitAI] Downloading: {download_url}")
        with requests.get(download_url, headers=headers, stream=True) as r:
            if r.status_code == 401:
                raise RuntimeError(
                    "CivitAI API error: 401 Unauthorized during download.\n\n"
                    "You must enter a valid API key, which you can get from your account settings on Civitai."
                )
            r.raise_for_status()

            with open(target_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"[CivitAI] Saved to: {target_path}")
        return ()
