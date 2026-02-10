import os
import json
import time
import base64
import tempfile
from io import BytesIO

from PIL import Image
import torch
import numpy as np


p = os.path.dirname(os.path.realpath(__file__))

CONFIG_FILENAME = "config.json"
CACHE_TTL_SECONDS = 48 * 3600  # Files API: 48h


def get_config():
    try:
        config_path = os.path.join(p, CONFIG_FILENAME)
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(config: dict):
    try:
        config_path = os.path.join(p, CONFIG_FILENAME)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    except Exception:
        pass


class NanoBananaUnified:
    """
    Unified ComfyUI node:
      - Model dropdown: gemini-2.5-flash-image / gemini-3-pro-image-preview
      - Keeps: Files API upload + cache, 5 refs, operations, aspect ratio, character consistency
      - Keeps (Gemini 3 node): seed/top_p/top_k/candidatecount, best-effort config builder, streaming log
      - Keeps (Gemini 2.5 node): ref upload format PNG/WEBP/JPEG/AUTO + quality settings
    """

    def __init__(self, apikey=None):
        envkey = os.environ.get("GEMINIAPIKEY")
        placeholders = {"tokenhere", "placetokenhere", "yourapikey", "apikeyhere", "enteryourkey", ""}
        if envkey and envkey.lower().strip() not in placeholders:
            self.apikey = envkey.strip()
        else:
            self.apikey = (apikey or "").strip() if isinstance(apikey, str) else apikey

        cfg = get_config()
        if not self.apikey:
            self.apikey = (cfg.get("GEMINIAPIKEY") or "").strip()

        self.filecache = cfg.get("filecache", {}) or {}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # keep behavior similar to the Gemini-3 node: force re-run
        return time.time_ns()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ([
                    "gemini-2.5-flash-image",
                    "gemini-3-pro-image-preview",
                ], {
                    "default": "gemini-2.5-flash-image",
                    "tooltip": "Choose the Gemini image model. 2.5 uses non-stream API path; 3 uses streaming path + advanced sampling knobs."
                }),
                "prompt": ("STRING", {
                    "forceInput": True,
                    "multiline": True,
                    "default": "",
                    "tooltip": "Describe what you want to generate or edit. Structured prompt markers are supported (IDENTITYLOCK/COREPROMPT/etc.)."
                }),
                "operation": (["generate", "edit", "styletransfer", "objectinsertion"], {
                    "default": "generate",
                    "tooltip": "Operation type."
                }),
            },
            "optional": {
                "referenceimage1": ("IMAGE", {"forceInput": False}),
                "referenceimage2": ("IMAGE", {"forceInput": False}),
                "referenceimage3": ("IMAGE", {"forceInput": False}),
                "referenceimage4": ("IMAGE", {"forceInput": False}),
                "referenceimage5": ("IMAGE", {"forceInput": False}),

                "apikey": ("STRING", {
                    "default": "",
                    "tooltip": "Gemini API key (paid tier typically required). If empty, will use env GEMINIAPIKEY or config.json."
                }),

                "batchcount": ("INT", {
                    "default": 1, "min": 1, "max": 4, "step": 1,
                    "tooltip": "How many calls/batches to run. Cost scales with batches."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Creativity."
                }),

                # Gemini-3 style sampling knobs (kept even if model=2.5; they will be ignored)
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 2147483647, "step": 1,
                    "tooltip": "Best-effort random seed (mainly used for gemini-3-pro-image-preview)."
                }),
                "topp": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Top-p nucleus sampling (best-effort; mainly for gemini-3-pro-image-preview)."
                }),
                "topk": ("INT", {
                    "default": 20, "min": 0, "max": 200, "step": 1,
                    "tooltip": "Top-k sampling (best-effort; mainly for gemini-3-pro-image-preview)."
                }),
                "candidatecount": ("INT", {
                    "default": 1, "min": 1, "max": 4, "step": 1,
                    "tooltip": "Candidate count (best-effort; mainly for gemini-3-pro-image-preview)."
                }),

                "quality": (["standard", "high"], {
                    "default": "high",
                    "tooltip": "Quality hint (used in prompt text)."
                }),
                "aspectratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {
                    "default": "1:1",
                    "tooltip": "Output aspect ratio."
                }),
                "imagesize": (["1K", "2K", "4K"], {
                    "default": "1K",
                    "tooltip": "Best-effort image size (mainly used for gemini-3-pro-image-preview)."
                }),

                "characterconsistency": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Ask the model to maintain identity/consistency from reference images."
                }),
                "enablesafety": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Safety filter toggle (best-effort; may be not wired depending on SDK/model)."
                }),

                # Gemini-2.5 node feature: upload format controls
                "refuploadformat": (["PNG", "WEBP", "JPEG", "AUTO"], {
                    "default": "PNG",
                    "tooltip": "Reference upload format (Files API). AUTO will fall back safely."
                }),
                "webpquality": ("INT", {
                    "default": 95, "min": 1, "max": 100, "step": 1,
                    "tooltip": "WEBP quality for reference uploads."
                }),
                "jpegquality": ("INT", {
                    "default": 95, "min": 1, "max": 100, "step": 1,
                    "tooltip": "JPEG quality for reference uploads."
                }),

                "minpromptchars": ("INT", {
                    "default": 10, "min": 0, "max": 500, "step": 1,
                    "tooltip": "Skip API call if prompt shorter than this. (2.5 node used 10; 3 node used 50; now configurable.)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generatedimages", "operationlog")
    FUNCTION = "nanobananagenerate"
    CATEGORY = "Nano Banana (Unified)"

    # ---------- image utils ----------

    def tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        t = tensor.detach().cpu()
        if len(t.shape) == 4:
            t = t[0]
        # assume 0..1 float
        t = t.clamp(0, 1)
        arr = (t.numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def create_placeholder_image(self, width: int = 512, height: int = 512) -> torch.Tensor:
        img = Image.new("RGB", (width, height), color=(100, 100, 100))
        try:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            draw.text((width // 2 - 60, height // 2 - 10), "Generation", fill=(255, 255, 255))
        except Exception:
            pass
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def fingerprint_tensor(self, image_tensor: torch.Tensor) -> str:
        pil = self.tensor_to_image(image_tensor[0] if (hasattr(image_tensor, "shape") and len(image_tensor.shape) == 4) else image_tensor)
        buf = BytesIO()
        pil.save(buf, format="PNG")
        raw = buf.getvalue()
        return base64.b64encode(raw).decode("utf-8")

    # ---------- cache helpers ----------

    def cache_get(self, cache_key: str):
        item = (self.filecache or {}).get(cache_key)
        if not item:
            return None
        uploaded_at = item.get("uploadedat", 0)
        if time.time() - uploaded_at > CACHE_TTL_SECONDS:
            return None
        refs = item.get("refs") or []
        if not isinstance(refs, list):
            return None
        # legacy form: list[str] names
        if refs and isinstance(refs[0], str):
            return [{"name": x, "mimetype": "image/png"} for x in refs]
        return refs

    def cache_set(self, cache_key: str, refs: list):
        cfg = get_config()
        cfg["GEMINIAPIKEY"] = self.apikey
        cfg.setdefault("filecache", {})
        cfg["filecache"][cache_key] = {"refs": refs, "uploadedat": time.time()}
        save_config(cfg)
        self.filecache = cfg.get("filecache", {}) or {}

    # ---------- Files API helpers ----------

    def _encode_ref(self, image_tensor: torch.Tensor, idx: int, refuploadformat: str, webpquality: int, jpegquality: int):
        pil = self.tensor_to_image(image_tensor[0] if (hasattr(image_tensor, "shape") and len(image_tensor.shape) == 4) else image_tensor)

        def make_buf(fmt: str, ext: str, mime: str, **savekwargs):
            buf = BytesIO()
            pil.save(buf, format=fmt, **savekwargs)
            buf.seek(0)
            buf.name = f"ref{idx}.{ext}"
            return buf, mime, ext

        fmt = (refuploadformat or "PNG").upper().strip()
        if fmt == "AUTO":
            fmt = "PNG"

        if fmt == "PNG":
            return make_buf("PNG", "png", "image/png")
        if fmt == "WEBP":
            return make_buf("WEBP", "webp", "image/webp", quality=int(webpquality), method=6)
        if fmt == "JPEG":
            return make_buf("JPEG", "jpg", "image/jpeg", quality=int(jpegquality))

        # fallback
        try:
            return make_buf("PNG", "png", "image/png")
        except Exception:
            return make_buf("JPEG", "jpg", "image/jpeg", quality=int(jpegquality))

    def _upload_file_strong(self, client, buf: BytesIO, mimetype: str, ext: str):
        # try: upload(fileobj, mimetype=...)
        try:
            return client.files.upload(file=buf, mime_type=mimetype)
        except TypeError:
            pass
        except Exception:
            pass

        # try: upload(fileobj)
        try:
            return client.files.upload(file=buf)
        except TypeError:
            pass
        except Exception:
            pass

        # fallback: write temp file then upload(path)
        tmp_path = None
        try:
            suffix = f".{ext}" if ext else ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                tmp_path = f.name
                f.write(buf.getvalue())
            try:
                return client.files.upload(file=tmp_path)
            except TypeError:
                return client.files.upload(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _wait_active_and_get_uri(self, client, filename: str, timeouts: int = 60):
        deadline = time.time() + timeouts
        last_state = None
        last_uri = None
        last_name = filename
        while time.time() < deadline:
            info = client.files.get(name=filename)
            state = getattr(info, "state", None)
            state_name = getattr(state, "name", state) if state else "UNKNOWN"
            last_state = state_name
            last_uri = getattr(info, "uri", None)
            last_name = getattr(info, "name", filename)
            if state_name == "ACTIVE":
                return last_name, last_uri, state_name
            if state_name == "FAILED":
                raise Exception(f"File upload failed for {filename}")
            time.sleep(2)
        return last_name, last_uri, last_state

    def _resolve_refs(self, client, refs: list):
        resolved = []
        for r in refs or []:
            name = r.get("name")
            mimetype = r.get("mimetype") or "image/png"
            uri = r.get("uri")
            if not name:
                continue
            if not uri:
                try:
                    active_name, active_uri, _ = self._wait_active_and_get_uri(client, name, timeouts=30)
                    name = active_name or name
                    uri = active_uri or uri
                except Exception:
                    continue
            resolved.append({"name": name, "uri": uri, "mimetype": mimetype})
        return resolved

    def upload_reference_images(self, refuploadformat: str, webpquality: int, jpegquality: int, *ref_images):
        images = [img for img in ref_images if img is not None]
        if not images or not self.apikey:
            return []

        # build cache key
        fingerprints = [self.fingerprint_tensor(img) for img in images]
        cache_key = "refset:" + (refuploadformat or "PNG") + ":" + ".".join(fingerprints)
        cached = self.cache_get(cache_key)
        if cached:
            return cached

        try:
            from google import genai
        except Exception:
            return []

        client = genai.Client(api_key=self.apikey)

        refs = []
        for idx, img in enumerate(images, start=1):
            buf, mimetype, ext = self._encode_ref(img, idx, refuploadformat, webpquality, jpegquality)
            uploaded = self._upload_file_strong(client, buf, mimetype=mimetype, ext=ext)
            name = getattr(uploaded, "name", None)
            if not name:
                continue
            active_name, active_uri, _ = self._wait_active_and_get_uri(client, name, timeouts=60)
            refs.append({"name": active_name, "uri": active_uri, "mimetype": mimetype})

        if refs:
            self.cache_set(cache_key, refs)
        return refs

    # ---------- prompt helpers ----------

    def build_prompt_for_operation(self, prompt: str, operation: str, has_references: bool, aspectratio: str, characterconsistency: bool):
        def is_structured_prompt(s: str) -> bool:
            if not isinstance(s, str):
                return False
            t = s.strip()
            markers = [
                "IDENTITYLOCK", "COREPROMPT", "STYLEMOOD", "SCENE", "HARDRULES",
                "NEGATIVECONSTRAINTS", "NEGATIVE"
            ]
            return any(m in t for m in markers)

        prompt = (prompt or "").strip()
        if not prompt:
            return "Error Empty prompt"

        # structured prompt: pass through (but still add operation wrapper where needed)
        if is_structured_prompt(prompt):
            if operation == "generate":
                return prompt
            if operation == "edit":
                if not has_references:
                    return "Error Edit operation requires reference images"
                return "Edit the provided reference images according to the instructions below.\n\n" + prompt
            if operation == "styletransfer":
                if not has_references:
                    return "Error Style transfer requires reference images"
                return "Apply the style from the reference images while following the instructions below.\n\n" + prompt
            if operation == "objectinsertion":
                if not has_references:
                    return "Error Object insertion requires reference images"
                return "Insert/blend requested elements into the reference images following the instructions below.\n\n" + prompt
            return prompt

        aspect_instructions = {
            "1:1": "square format",
            "16:9": "widescreen landscape format",
            "9:16": "portrait format",
            "4:3": "standard landscape format",
            "3:4": "standard portrait format",
        }
        fmt = aspect_instructions.get(aspectratio, "square format")
        basequality = "Generate a high-quality, photorealistic image"

        if operation == "generate":
            if has_references:
                final = f"{basequality}. {prompt}. Output in {fmt}."
            else:
                final = f"{basequality} of {prompt}. Output in {fmt}."
        elif operation == "edit":
            if not has_references:
                return "Error Edit operation requires reference images"
            final = f"Edit the provided reference images. {prompt}. Maintain the original composition and quality while making the requested changes."
        elif operation == "styletransfer":
            if not has_references:
                return "Error Style transfer requires reference images"
            final = f"Apply the style from the reference images to create: {prompt}. Blend the stylistic elements naturally. Output in {fmt}."
        elif operation == "objectinsertion":
            if not has_references:
                return "Error Object insertion requires reference images"
            final = f"Insert or blend the following into the reference images: {prompt}. Ensure natural lighting, shadows, and perspective. Output in {fmt}."
        else:
            final = f"{basequality} of {prompt}. Output in {fmt}."

        if characterconsistency and has_references:
            final += " Maintain character consistency and visual identity from the reference images."
        return final

    # ---------- config builders ----------

    def build_config_best_effort(self, types_mod, temperature, imagesize, aspectratio, seed, topp, topk, candidatecount):
        basekwargs = {
            "temperature": float(temperature),
            "response_modalities": ["IMAGE", "TEXT"],
        }

        imgcfg = None
        try:
            imgcfg = types_mod.ImageConfig(image_size=imagesize, aspect_ratio=str(aspectratio).strip())
        except Exception:
            try:
                imgcfg = types_mod.ImageConfig(image_size=imagesize)
            except Exception:
                imgcfg = None

        if imgcfg is not None:
            basekwargs["image_config"] = imgcfg

        # try multiple signatures for SDK compatibility
        attempts = [
            dict(basekwargs, seed=int(seed), top_p=float(topp), top_k=int(topk), candidate_count=int(candidatecount)),
            dict(basekwargs, seed=int(seed), top_p=float(topp), top_k=int(topk)),
            dict(basekwargs, seed=int(seed), top_p=float(topp)),
            dict(basekwargs, seed=int(seed)),
            dict(basekwargs),
        ]

        for kw in attempts:
            try:
                return types_mod.GenerateContentConfig(**kw)
            except Exception:
                continue

        return types_mod.GenerateContentConfig(temperature=float(temperature), response_modalities=["IMAGE", "TEXT"])

    def build_config_25(self, types_mod, temperature, aspectratio):
        # minimal, stable config for gemini-2.5-flash-image path
        try:
            imgcfg = types_mod.ImageConfig(aspect_ratio=str(aspectratio).strip())
        except Exception:
            imgcfg = None
        kwargs = {"temperature": float(temperature), "response_modalities": ["IMAGE"]}
        if imgcfg is not None:
            kwargs["image_config"] = imgcfg
        try:
            return types_mod.GenerateContentConfig(**kwargs)
        except Exception:
            return types_mod.GenerateContentConfig(temperature=float(temperature), response_modalities=["IMAGE"])

    # ---------- model callers ----------

    def call_model_25(self, prompt, refs, temperature, batchcount, aspectratio):
        try:
            from google import genai
            from google.genai import types
        except Exception:
            return [], "ERROR google-genai not installed. Run: pip install -U google-genai"

        if not self.apikey:
            return [], "ERROR Missing GEMINIAPIKEY"

        client = genai.Client(api_key=self.apikey)
        resolved = self._resolve_refs(client, refs)

        parts = []
        try:
            parts.append(types.Part.from_text(text=prompt))
        except Exception:
            parts.append(types.Part(text=prompt))

        for r in resolved:
            if r.get("uri"):
                parts.append(types.Part.from_uri(file_uri=r["uri"], mime_type=r.get("mimetype") or "image/png"))

        config = self.build_config_25(types, temperature=temperature, aspectratio=aspectratio)

        all_tensors = []
        err_messages = []
        for i in range(int(batchcount)):
            try:
                resp = client.models.generate_content(
                    model="gemini-2.5-flash-image",
                    contents=types.Content(role="user", parts=parts),
                    config=config,
                )
                cands = getattr(resp, "candidates", None)
                if not cands:
                    err_messages.append(f"Batch {i+1}: no candidates")
                    continue

                image_found = False
                for part in cands[0].content.parts:
                    inlinedata = getattr(part, "inlinedata", None) or getattr(part, "inline_data", None)
                    if inlinedata is None:
                        continue
                    data = getattr(inlinedata, "data", None)
                    if data is None:
                        continue

                    # 2.5 path may return base64 string; be tolerant
                    if isinstance(data, str):
                        try:
                            data = base64.b64decode(data)
                        except Exception:
                            data = data.encode("utf-8", errors="ignore")

                    img = Image.open(BytesIO(data)).convert("RGB")
                    arr = (np.array(img).astype(np.float32) / 255.0)
                    all_tensors.append(torch.from_numpy(arr)[None, ...])
                    image_found = True
                    break

                if not image_found:
                    err_messages.append(f"Batch {i+1}: no image data")
            except Exception as e:
                err_messages.append(f"Batch {i+1}: {str(e)}")

        if all_tensors:
            return all_tensors, "\n".join(err_messages).strip()
        return [], "\n".join(err_messages).strip() or "ERROR no images returned"

    def call_model_3_stream(self, prompt, refs, temperature, batchcount, enablesafety, seed, topp, topk, candidatecount, imagesize, aspectratio, modelname):
        try:
            from google import genai
            from google.genai import types
        except Exception:
            return [], "ERROR google-genai not installed. Run: pip install -U google-genai"

        if not self.apikey:
            return [], "ERROR Missing GEMINIAPIKEY"

        client = genai.Client(api_key=self.apikey)
        resolved = self._resolve_refs(client, refs)

        parts = []
        for r in resolved:
            if r.get("uri"):
                parts.append(types.Part.from_uri(file_uri=r["uri"], mime_type=r.get("mimetype") or "image/png"))

        try:
            parts.append(types.Part.from_text(text=prompt))
        except Exception:
            parts.append(types.Part(text=prompt))

        contents = types.Content(role="user", parts=parts)

        config = self.build_config_best_effort(
            types_mod=types,
            temperature=temperature,
            imagesize=imagesize,
            aspectratio=aspectratio,
            seed=seed,
            topp=topp,
            topk=topk,
            candidatecount=candidatecount,
        )

        oplog = []
        all_image_bytes = []

        for i in range(int(batchcount)):
            oplog.append(f"Batch {i+1}: calling model stream...")
            try:
                for chunk in client.models.generate_content_stream(
                    model=modelname,
                    contents=contents,
                    config=config,
                ):
                    cands = getattr(chunk, "candidates", None)
                    if not cands:
                        continue
                    for cand in cands:
                        content = getattr(cand, "content", None)
                        if content is None:
                            continue
                        candparts = getattr(content, "parts", None)
                        if not candparts:
                            continue
                        for part in candparts:
                            inlinedata = getattr(part, "inlinedata", None) or getattr(part, "inline_data", None)
                            if inlinedata is not None and getattr(inlinedata, "data", None) is not None:
                                all_image_bytes.append(inlinedata.data)
                            txt = getattr(part, "text", None)
                            if txt:
                                oplog.append(txt)
            except Exception as e:
                oplog.append(f"Batch {i+1}: API error {str(e)}")

        tensors = []
        for b in all_image_bytes:
            try:
                if isinstance(b, str):
                    try:
                        b = base64.b64decode(b)
                    except Exception:
                        b = b.encode("utf-8", errors="ignore")
                img = Image.open(BytesIO(b))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                arr = (np.array(img).astype(np.float32) / 255.0)
                tensors.append(torch.from_numpy(arr)[None, ...])
            except Exception as e:
                oplog.append(f"Error processing image: {str(e)}")

        oplog.append(f"Total images: {len(tensors)}")
        return tensors, "\n".join(oplog)

    # ---------- main entry ----------

    def nanobananagenerate(
        self,
        model,
        prompt,
        operation,
        referenceimage1=None,
        referenceimage2=None,
        referenceimage3=None,
        referenceimage4=None,
        referenceimage5=None,
        apikey="",
        batchcount=1,
        temperature=0.2,
        seed=42,
        topp=0.9,
        topk=20,
        candidatecount=1,
        quality="high",
        aspectratio="1:1",
        imagesize="1K",
        characterconsistency=True,
        enablesafety=True,
        refuploadformat="PNG",
        webpquality=95,
        jpegquality=95,
        minpromptchars=10,
    ):
        # API key resolution
        if isinstance(apikey, str) and apikey.strip():
            self.apikey = apikey.strip()
            cfg = get_config()
            cfg["GEMINIAPIKEY"] = self.apikey
            cfg.setdefault("filecache", cfg.get("filecache", {}) or {})
            save_config(cfg)
            self.filecache = (cfg.get("filecache", {}) or {})

        if not self.apikey:
            return (self.create_placeholder_image(),
                    "NANO BANANA ERROR No API key provided! Set GEMINIAPIKEY env or input apikey.")

        prompt_clean = (prompt or "").strip()
        if len(prompt_clean) < int(minpromptchars):
            msg = f"Skip API call: prompt too short (len={len(prompt_clean)} < {int(minpromptchars)})."
            return self.create_placeholder_image(), msg

        # refs + prompt
        refs = self.upload_reference_images(
            refuploadformat, int(webpquality), int(jpegquality),
            referenceimage1, referenceimage2, referenceimage3, referenceimage4, referenceimage5
        )
        has_refs = len(refs) > 0

        finalprompt = self.build_prompt_for_operation(
            prompt_clean, operation, has_refs, aspectratio, bool(characterconsistency)
        )
        if "Error" in finalprompt:
            return self.create_placeholder_image(), finalprompt

        # op log header
        op = []
        op.append("NANO BANANA UNIFIED OPERATION LOG")
        op.append(f"Model: {model}")
        op.append(f"Operation: {str(operation).upper()}")
        op.append(f"Reference Images: {len(refs)}")
        op.append(f"Batch Count: {batchcount}")
        op.append(f"Temperature: {temperature}")
        op.append(f"Seed: {seed} (best-effort; mainly gemini-3)")
        op.append(f"TopP: {topp} (best-effort; mainly gemini-3)")
        op.append(f"TopK: {topk} (best-effort; mainly gemini-3)")
        op.append(f"CandidateCount: {candidatecount} (best-effort; mainly gemini-3)")
        op.append(f"Quality hint: {quality}")
        op.append(f"Aspect Ratio: {aspectratio}")
        op.append(f"Image Size: {imagesize} (best-effort; mainly gemini-3)")
        op.append(f"Character Consistency: {characterconsistency}")
        op.append(f"Safety Filters: {enablesafety} (best-effort; may be not wired)")
        op.append(f"Ref Upload Format: {refuploadformat} (WEBP/JPEG quality applied if used)")
        op.append(f"Prompt (preview): {finalprompt[:300]}")

        # dispatch
        if model == "gemini-2.5-flash-image":
            # note: advanced sampling knobs are not used in this path
            tensors, log = self.call_model_25(
                prompt=finalprompt,
                refs=refs,
                temperature=temperature,
                batchcount=batchcount,
                aspectratio=aspectratio,
            )
            if log:
                op.append(log)
        else:
            tensors, log = self.call_model_3_stream(
                prompt=finalprompt,
                refs=refs,
                temperature=temperature,
                batchcount=batchcount,
                enablesafety=enablesafety,
                seed=seed,
                topp=topp,
                topk=topk,
                candidatecount=candidatecount,
                imagesize=imagesize,
                aspectratio=aspectratio,
                modelname=model,
            )
            if log:
                op.append(log)

        if tensors:
            combined = torch.cat(tensors, dim=0)
            op.append(f"Successfully generated {combined.shape[0]} image(s).")
            return combined, "\n".join(op)

        op.append("No images were generated. Check the log above for details.")
        return self.create_placeholder_image(), "\n".join(op)


NODE_CLASS_MAPPINGS = {
    "NanoBananaUnified": NanoBananaUnified
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaUnified": "Nano Banana (Gemini 2.5 / Gemini 3) Unified"
}
