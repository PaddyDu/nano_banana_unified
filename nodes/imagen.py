"""NanoBananaImagen — Vertex AI Imagen image generation node."""

import os
import time
import tempfile
import hashlib
from io import BytesIO

from PIL import Image
import torch
import numpy as np

from .utils import get_config, save_config

# Models that use recontext_image API instead of edit_image
RECONTEXT_MODELS = {"virtual-try-on-001", "imagen-product-recontext-preview-06-30"}


class NanoBananaImagen:
    """
    ComfyUI node: generate person images via Google Vertex AI Imagen.

    Supports:
    - edit_image + SubjectReferenceImage(PERSON) → person in new scene
    - edit_image + ContentReferenceImage → Imagen4 quality scene generation
    - recontext_image + virtual-try-on-001 → person wearing specific outfit
    - recontext_image + product-recontext → product in new scene (requires allowlist)

    Auth: gcloud ADC (gcloud auth application-default login)
    """

    def __init__(self):
        pass

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
                    "tooltip": "描述场景和服装（virtual-try-on 模式下会被忽略）",
                }),
                "gcpprojectid": ("STRING", {"default": "", "tooltip": "GCP Project ID"}),
            },
            "optional": {
                "referenceimage1": ("IMAGE", {"forceInput": True,  "tooltip": "参考图 1（必填）"}),
                "referenceimage2": ("IMAGE", {"forceInput": False, "tooltip": "参考图 2 / virtual-try-on 服装图"}),
                "referenceimage3": ("IMAGE", {"forceInput": False, "tooltip": "参考图 3"}),
                "gcslocation":     ("STRING", {"default": "us-central1", "tooltip": "GCP Region"}),
                "gcsbucket":       ("STRING", {"default": "", "tooltip": "GCS Bucket（可选，用于缓存参考图 URI）"}),
                "aspectratio":     (["1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "9:16"}),
                "imagenmodel": ([
                    "imagen-3.0-capability-001",
                    "imagen-4.0-ingredients-preview",
                    "virtual-try-on-001",
                    "imagen-product-recontext-preview-06-30",
                ], {
                    "default": "imagen-3.0-capability-001",
                    "tooltip": (
                        "capability-001: SubjectReferenceImage(PERSON) 人物一致性 | "
                        "ingredients: ContentReferenceImage Imagen4画质 | "
                        "virtual-try-on: ref1=人物 ref2=服装→穿搭效果 | "
                        "product-recontext: 产品置场景（需申请权限）"
                    ),
                }),
                "numberofimages":  ([1, 2, 3, 4], {"default": 1, "tooltip": "每次生成图片数量（1-4）"}),
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

    def _tensor_hash(self, tensor: torch.Tensor) -> str:
        t = tensor.detach().cpu()
        if t.ndim == 4:
            t = t[0]
        return hashlib.md5(t.numpy().tobytes()).hexdigest()[:16]

    def _tensor_to_png_bytes(self, tensor: torch.Tensor) -> bytes:
        buf = BytesIO()
        self._tensor_to_pil(tensor).save(buf, format="PNG")
        return buf.getvalue()

    # ─────────────────────────── GCS cache ──────────────────────────────────

    def _gcs_cache_get(self, cache_key: str):
        cfg = get_config()
        return (cfg.get("imagencache") or {}).get(cache_key, {}).get("gcs_uri")

    def _gcs_cache_set(self, cache_key: str, gcs_uri: str):
        cfg = get_config()
        cfg.setdefault("imagencache", {})
        cfg["imagencache"][cache_key] = {"gcs_uri": gcs_uri, "uploadedat": time.time()}
        save_config(cfg)

    def _upload_to_gcs(self, tensor: torch.Tensor, bucket_name: str, project_id: str):
        h = self._tensor_hash(tensor)
        cache_key = f"ref:{h}"
        cached = self._gcs_cache_get(cache_key)
        if cached:
            return cached
        try:
            from google.cloud import storage
            buf = BytesIO()
            self._tensor_to_pil(tensor).save(buf, format="PNG")
            buf.seek(0)
            client = storage.Client(project=project_id)
            bucket = client.bucket(bucket_name.strip())
            blob_name = f"nano_banana_refs/{h}.png"
            blob = bucket.blob(blob_name)
            if not blob.exists():
                blob.upload_from_file(buf, content_type="image/png")
            gcs_uri = f"gs://{bucket_name.strip()}/{blob_name}"
            self._gcs_cache_set(cache_key, gcs_uri)
            return gcs_uri
        except Exception:
            return None

    # ──────────────────────────────── main ──────────────────────────────────

    def generate(self, prompt, gcpprojectid,
                 referenceimage1=None, referenceimage2=None, referenceimage3=None,
                 gcslocation="us-central1", gcsbucket="", aspectratio="9:16",
                 imagenmodel="imagen-3.0-capability-001", numberofimages=1):

        if not gcpprojectid.strip():
            return self._placeholder(), "ERROR: 请提供 GCP Project ID"

        ref_images = [img for img in [referenceimage1, referenceimage2, referenceimage3] if img is not None]
        if not ref_images:
            return self._placeholder(), "ERROR: 请提供至少 1 张参考图"

        # Ref image limits per model
        is_recontext = imagenmodel in RECONTEXT_MODELS
        if imagenmodel == "virtual-try-on-001":
            ref_images = ref_images[:2]   # person + clothing
        elif is_recontext:
            ref_images = ref_images[:3]   # product: up to 3 views
        else:
            ref_limit = 3 if aspectratio == "1:1" else 2
            ref_images = ref_images[:ref_limit]

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return self._placeholder(), "ERROR: 请安装 google-genai: pip install -U google-genai"

        os.environ["GOOGLE_CLOUD_PROJECT"] = gcpprojectid.strip()

        try:
            client = genai.Client(
                vertexai=True,
                project=gcpprojectid.strip(),
                location=gcslocation.strip(),
            )
        except Exception as e:
            return self._placeholder(), f"ERROR: Vertex AI 客户端初始化失败 - {e}"

        use_gcs = bool(gcsbucket.strip())
        log_mode = "GCS" if use_gcs else "bytes"
        n = max(1, min(int(numberofimages), 4))
        response = None
        ref_type_log = ""

        def make_image(tensor):
            """Tensor → types.Image, with optional GCS caching."""
            if use_gcs:
                uri = self._upload_to_gcs(tensor, gcsbucket, gcpprojectid)
                if uri:
                    return types.Image(gcs_uri=uri)
            return types.Image(
                image_bytes=self._tensor_to_png_bytes(tensor),
                mime_type="image/png",
            )

        try:
            # ── recontext_image path ──────────────────────────────────────
            if is_recontext:
                if imagenmodel == "virtual-try-on-001":
                    if len(ref_images) < 2:
                        return self._placeholder(), (
                            "ERROR: virtual-try-on 需要 2 张图: "
                            "referenceimage1=人物照, referenceimage2=服装图"
                        )
                    source = types.RecontextImageSource(
                        person_image=make_image(ref_images[0]),
                        product_images=[types.ProductImage(product_image=make_image(ref_images[1]))],
                    )
                    ref_type_log = "VirtualTryOn"
                else:
                    source = types.RecontextImageSource(
                        prompt=prompt,
                        product_images=[
                            types.ProductImage(product_image=make_image(img))
                            for img in ref_images
                        ],
                    )
                    ref_type_log = f"ProductRecontext({len(ref_images)}张)"

                recontext_config = types.RecontextImageConfig(
                    number_of_images=n,
                    output_mime_type="image/jpeg",
                    person_generation=types.PersonGeneration.ALLOW_ALL,
                )
                # Auto-retry on SSL/network errors (up to 3x)
                last_err = None
                for attempt in range(3):
                    try:
                        response = client.models.recontext_image(
                            model=imagenmodel,
                            source=source,
                            config=recontext_config,
                        )
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        if any(k in str(e).lower() for k in ("ssl", "eof", "connection", "timeout", "reset")):
                            time.sleep(2 * (attempt + 1))
                            continue
                        break
                if last_err is not None:
                    return self._placeholder(), f"ERROR: 图片生成失败 - {last_err}"

            # ── edit_image path ───────────────────────────────────────────
            else:
                reference_images = []
                for idx, img in enumerate(ref_images, start=1):
                    ref_img = make_image(img)
                    if imagenmodel == "imagen-4.0-ingredients-preview":
                        ref = types.ContentReferenceImage(reference_id=idx, reference_image=ref_img)
                    else:
                        ref = types.SubjectReferenceImage(
                            reference_id=idx,
                            reference_image=ref_img,
                            config=types.SubjectReferenceConfig(
                                subject_type=types.SubjectReferenceType.SUBJECT_TYPE_PERSON,
                            ),
                        )
                    reference_images.append(ref)

                ref_ids = " ".join(f"[{i}]" for i in range(1, len(ref_images) + 1))
                if imagenmodel == "imagen-4.0-ingredients-preview":
                    full_prompt = (
                        f"Person {ref_ids} in the following scene: {prompt}. "
                        "Use the face and appearance of the person from the reference images."
                    )
                    ref_type_log = "ContentRef(Imagen4)"
                else:
                    full_prompt = (
                        f"Generate a high-quality photo-realistic image of the person {ref_ids}. "
                        f"The person should be: {prompt}. "
                        "Maintain the exact facial identity, skin tone and features from the reference."
                    )
                    ref_type_log = "SubjectRef(PERSON)"

                try:
                    response = client.models.edit_image(
                        model=imagenmodel,
                        prompt=full_prompt,
                        reference_images=reference_images,
                        config=types.EditImageConfig(
                            number_of_images=n,
                            aspect_ratio=aspectratio,
                            include_rai_reason=True,
                        ),
                    )
                except Exception as e:
                    return self._placeholder(), f"ERROR: 图片生成失败 - {e}"

        finally:
            pass

        if response is None or not response.generated_images:
            return self._placeholder(), "ERROR: 未生成图片"

        frames = []
        for gen_img in response.generated_images:
            raw = getattr(gen_img, "image", None)
            if not raw:
                continue
            img_data = getattr(raw, "image_bytes", None)
            if not img_data:
                continue
            try:
                pil = Image.open(BytesIO(img_data)).convert("RGB")
                frames.append(np.array(pil).astype(np.float32) / 255.0)
            except Exception:
                pass

        if not frames:
            return self._placeholder(), "ERROR: 无法获取图片数据"

        batch = torch.from_numpy(np.stack(frames, axis=0))
        return batch, (
            f"✅ 生成 {len(frames)} 张 | 模型: {imagenmodel} | "
            f"参考图: {len(ref_images)}张 | {log_mode} | {ref_type_log}"
        )
