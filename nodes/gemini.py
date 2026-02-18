"""NanoBananaGemini — Gemini multimodal image generation node."""

import os
import time
import base64
from io import BytesIO

from PIL import Image
import torch
import numpy as np

from .utils import get_config, save_config, CACHE_TTL_SECONDS


class NanoBananaGemini:
    """
    ComfyUI node: generate person images via Google Gemini API.
    Upload 1-3 face reference images → generate consistent person in new scene/outfit.
    Auth: Gemini API Key (config.json / env GEMINIAPIKEY / node widget)
    """

    def __init__(self):
        self.apikey = None
        self.filecache = {}
        self._load_apikey()

    def _load_apikey(self):
        envkey = os.environ.get("GEMINIAPIKEY")
        placeholders = {"tokenhere", "placetokenhere", "yourapikey", "apikeyhere", "enteryourkey", ""}
        if envkey and envkey.lower().strip() not in placeholders:
            self.apikey = envkey.strip()
        else:
            cfg = get_config()
            self.apikey = (cfg.get("GEMINIAPIKEY") or "").strip()
            self.filecache = cfg.get("filecache", {}) or {}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "forceInput": True,
                    "multiline": True,
                    "default": "A person dancing gracefully in a modern studio, professional photography",
                    "tooltip": "描述场景和服装",
                }),
            },
            "optional": {
                "referenceimage1": ("IMAGE", {"forceInput": True,  "tooltip": "人脸参考图 1（必填）"}),
                "referenceimage2": ("IMAGE", {"forceInput": False, "tooltip": "人脸参考图 2（可选）"}),
                "referenceimage3": ("IMAGE", {"forceInput": False, "tooltip": "人脸参考图 3（可选）"}),
                "apikey":          ("STRING", {"default": "", "tooltip": "Gemini API Key（留空则读 config.json / 环境变量）"}),
                "aspectratio":     (["1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "9:16"}),
                "model":           (["gemini-2.5-flash-image", "gemini-3-pro-image-preview"], {"default": "gemini-2.5-flash-image"}),
                "numberofimages":  ([1, 2, 3, 4], {"default": 1, "tooltip": "每次请求生成的图片数量（每张独立请求）"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "log")
    FUNCTION = "generate"
    CATEGORY = "Nano Banana"

    # ──────────────────────────────── helpers ────────────────────────────────

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        t = tensor.detach().cpu()
        if t.ndim == 4:
            t = t[0]
        arr = (t.clamp(0, 1).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def _placeholder(self) -> torch.Tensor:
        arr = np.zeros((512, 512, 3), dtype=np.float32)
        return torch.from_numpy(arr).unsqueeze(0)

    def _fingerprint(self, tensor: torch.Tensor) -> str:
        buf = BytesIO()
        self._tensor_to_pil(tensor).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _cache_get(self, key: str):
        item = (self.filecache or {}).get(key)
        if not item:
            return None
        if time.time() - item.get("uploadedat", 0) > CACHE_TTL_SECONDS:
            return None
        return item.get("refs")

    def _cache_set(self, key: str, refs: list):
        cfg = get_config()
        cfg["GEMINIAPIKEY"] = self.apikey
        cfg.setdefault("filecache", {})
        cfg["filecache"][key] = {"refs": refs, "uploadedat": time.time()}
        save_config(cfg)
        self.filecache = cfg.get("filecache", {})

    # ─────────────────────────── Files API upload ────────────────────────────

    def _upload_images(self, *images):
        images = [img for img in images if img is not None]
        if not images or not self.apikey:
            return []

        cache_key = "refs:" + ".".join(self._fingerprint(img) for img in images)
        cached = self._cache_get(cache_key)
        if cached:
            return cached

        try:
            from google import genai
            from google.genai import types as _gtypes
        except ImportError:
            return []

        client = genai.Client(api_key=self.apikey)
        refs = []

        for idx, img in enumerate(images, start=1):
            buf = BytesIO()
            self._tensor_to_pil(img).save(buf, format="PNG")
            buf.seek(0)
            buf.name = f"face{idx}.png"

            # google-genai 1.63.0+: mime_type via UploadFileConfig
            uploaded = None
            try:
                upload_cfg = _gtypes.UploadFileConfig(
                    mime_type="image/png",
                    display_name=f"face{idx}.png",
                )
                buf.seek(0)
                uploaded = client.files.upload(file=buf, config=upload_cfg)
            except Exception:
                try:
                    buf.seek(0)
                    uploaded = client.files.upload(file=buf, mime_type="image/png")  # older SDK
                except Exception:
                    pass

            if not uploaded:
                continue

            name = getattr(uploaded, "name", None)
            if not name:
                continue

            # Poll until ACTIVE
            deadline = time.time() + 60
            while time.time() < deadline:
                info = client.files.get(name=name)
                state = getattr(info, "state", None)
                state_name = getattr(state, "name", state) if state else "UNKNOWN"
                if state_name == "ACTIVE":
                    uri = getattr(info, "uri", None)
                    if uri:
                        refs.append({"name": name, "uri": uri, "mimetype": "image/png"})
                    break
                elif state_name == "FAILED":
                    break
                time.sleep(2)

        if refs:
            self._cache_set(cache_key, refs)
        return refs

    # ──────────────────────────────── main ──────────────────────────────────

    def generate(self, prompt,
                 referenceimage1=None, referenceimage2=None, referenceimage3=None,
                 apikey="", aspectratio="9:16", model="gemini-2.5-flash-image",
                 numberofimages=1):

        if isinstance(apikey, str) and apikey.strip():
            self.apikey = apikey.strip()

        if not self.apikey:
            return self._placeholder(), "ERROR: 请提供 Gemini API Key"

        ref_images = [img for img in [referenceimage1, referenceimage2, referenceimage3] if img is not None]
        if not ref_images:
            return self._placeholder(), "ERROR: 请提供至少 1 张人脸参考图"

        refs = self._upload_images(*ref_images)
        if not refs:
            return self._placeholder(), "ERROR: 参考图片上传失败"

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return self._placeholder(), "ERROR: 请安装 google-genai: pip install -U google-genai"

        client = genai.Client(api_key=self.apikey)

        full_prompt = (
            f"Generate a high-quality image of the SAME person shown in the reference images.\n"
            f"The person should be: {prompt}\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "- Maintain EXACT facial identity from reference images\n"
            "- Keep the same person's facial features, skin tone, and general appearance\n"
            "- Apply the new scene, clothing, and pose as described\n"
            "- Professional photography quality\n"
            f"- Aspect ratio: {aspectratio}"
        )

        parts = [
            types.Part.from_uri(file_uri=r["uri"], mime_type=r.get("mimetype", "image/png"))
            for r in refs if r.get("uri")
        ]
        parts.append(types.Part.from_text(text=full_prompt))

        # Gemini image models don't support candidate_count → loop N requests
        n = max(1, min(int(numberofimages), 4))
        try:
            gen_config = types.GenerateContentConfig(
                temperature=0.3,
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio=aspectratio),
            )
        except Exception:
            gen_config = types.GenerateContentConfig(
                temperature=0.3,
                response_modalities=["IMAGE"],
            )

        frames, errors = [], []
        for _ in range(n):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=types.Content(role="user", parts=parts),
                    config=gen_config,
                )
                cands = getattr(resp, "candidates", None) or []
                for part in getattr(getattr(cands[0], "content", None), "parts", None) or []:
                    inline = getattr(part, "inlinedata", None) or getattr(part, "inline_data", None)
                    if inline:
                        data = getattr(inline, "data", None)
                        if data:
                            if isinstance(data, str):
                                data = base64.b64decode(data)
                            try:
                                pil = Image.open(BytesIO(data)).convert("RGB")
                                frames.append(np.array(pil).astype(np.float32) / 255.0)
                            except Exception:
                                pass
            except Exception as e:
                errors.append(str(e))

        if not frames:
            return self._placeholder(), f"ERROR: API 调用失败 - {errors[0] if errors else '未知错误'}"

        batch = torch.from_numpy(np.stack(frames, axis=0))
        return batch, f"✅ 生成 {len(frames)} 张 | 模型: {model} | 参考图: {len(refs)}张"
