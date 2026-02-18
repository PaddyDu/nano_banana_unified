# 🍌 Nano Banana Unified

> **Generate consistent person images in ComfyUI using Google's latest AI models.**  
> Drop in 1-3 face photos → get the same person in any scene, outfit, or style.

Two powerful nodes powered by **Google Gemini** and **Vertex AI Imagen**:

| Node | Auth | Best For |
|------|------|----------|
| **NanoBananaGemini** | API Key | Quick prototyping, no GCP required |
| **NanoBananaImagen** | gcloud ADC | Production quality, virtual try-on, GCS caching |

---

## ✨ Features

- 🎭 **Person Identity Lock** — Feed 1-3 face reference images, get the same person anywhere
- 👗 **Virtual Try-On** — Person photo + clothing photo → wear the outfit (powered by `virtual-try-on-001`)
- 🖼️ **Multi-Image Batch** — Generate up to 4 images per request, returned as a batch tensor
- ⚡ **Smart Caching** — Gemini Files API URIs cached 48h; GCS URIs cached permanently
- 🔄 **Auto-Retry** — SSL/network errors retry automatically (up to 3×)
- 🤖 **4 Imagen Models** — Choose between quality vs. identity precision vs. outfit generation

### Supported Imagen Models

| Model | API | Input | Best For |
|-------|-----|-------|----------|
| `imagen-3.0-capability-001` | `edit_image` | 1-3 face photos + prompt | **Strongest person identity lock** |
| `imagen-4.0-ingredients-preview` | `edit_image` | 1-2 face photos + prompt | Best image quality (Imagen 4) |
| `virtual-try-on-001` ✨ | `recontext_image` | person photo + clothing photo | **Virtual outfit try-on** |
| `imagen-product-recontext-preview-06-30` | `recontext_image` | 1-3 product photos + prompt | Product in new scene *(requires allowlist)* |

---

## 📦 Installation

### 1. Clone

```bash
cd /path/to/ComfyUI/custom_nodes/
git clone https://github.com/PaddyDu/nano_banana_unified.git
```

### 2. Install Dependencies

> ⚠️ **Important**: Always install into ComfyUI's **parent** `.venv`, not a new local one.

```bash
/path/to/ComfyUI/.venv/bin/pip install google-genai google-cloud-storage
```

### 3. Restart ComfyUI

Nodes appear under **Nano Banana** category.

---

## 🔑 Authentication

### NanoBananaGemini — API Key

Get a free key at [Google AI Studio](https://aistudio.google.com/).

**Option A: `config.json`** (recommended)
```bash
cp config.example.json config.json
# Edit config.json → set GEMINIAPIKEY
```

**Option B: Environment variable**
```bash
export GEMINIAPIKEY="AIzaSy..."
```

**Option C: Node widget** — paste the key directly in the `apikey` field.

---

### NanoBananaImagen — Vertex AI ADC

No API key needed. Uses Google Cloud Application Default Credentials.

```bash
# Install gcloud CLI (macOS)
brew install google-cloud-sdk

# Two-step auth (both are required!)
gcloud auth login
gcloud auth application-default login

# Verify
gcloud auth application-default print-access-token
```

> ⚠️ **Gotcha**: `gcloud auth login` alone is **not enough**. You must also run `gcloud auth application-default login`.

---

## 🪣 GCS Bucket Setup (Optional but Recommended)

Without a bucket, reference images are sent as raw bytes every request (~100-500 KB/image).  
With a bucket, images are uploaded once and referenced forever by URI — much faster.

```bash
# Create bucket
gcloud storage buckets create gs://your-bucket \
  --project=YOUR_PROJECT_ID --location=us-central1 \
  --uniform-bucket-level-access

# Grant Imagen API read access (user type, NOT serviceAccount)
gcloud storage buckets add-iam-policy-binding gs://your-bucket \
  --member="user:cloud-lvm@prod.google.com" \
  --role="roles/storage.objectViewer"

# Grant yourself write access
gcloud storage buckets add-iam-policy-binding gs://your-bucket \
  --member="user:your-email@gmail.com" \
  --role="roles/storage.objectAdmin"
```

> ⚠️ **Gotcha**: The Imagen service reads GCS via `cloud-lvm@prod.google.com` — it's a **`user:`** type principal, not `serviceAccount:`.

---

## 🧭 Node Usage

### NanoBananaGemini

| Input | Description |
|-------|-------------|
| `prompt` | Scene / outfit description |
| `referenceimage1/2/3` | Face reference photos (1-3) |
| `model` | `gemini-2.5-flash-image` (fast) or `gemini-3-pro-image-preview` |
| `aspectratio` | Output aspect ratio |
| `numberofimages` | 1-4 images (each is a separate request) |

### NanoBananaImagen

| Input | Description |
|-------|-------------|
| `prompt` | Scene description *(ignored for virtual-try-on)* |
| `gcpprojectid` | Your GCP project ID |
| `referenceimage1` | Person face / person full-body (VTO) |
| `referenceimage2` | Extra face ref *or* clothing image (VTO) |
| `referenceimage3` | Extra face ref (not used in VTO) |
| `gcsbucket` | GCS bucket name (optional, enables caching) |
| `imagenmodel` | See model table above |
| `numberofimages` | 1-4 images per request |

**Virtual Try-On tip:**
- `referenceimage1` = full person photo
- `referenceimage2` = clothing flat-lay / product photo (not worn)
- `prompt` is automatically ignored

---

## 🗂️ Project Structure

```
nano_banana_unified/
├── __init__.py          # ComfyUI entry point (node registration)
├── nodes/
│   ├── gemini.py        # NanoBananaGemini implementation
│   ├── imagen.py        # NanoBananaImagen implementation
│   └── utils.py         # Shared: config I/O, constants
├── config.example.json  # API key template
└── requirements.txt     # Python dependencies
```

---

## 🐛 Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `No project ID could be determined` | Missing env var | Node sets `GOOGLE_CLOUD_PROJECT` automatically |
| `No uri or raw bytes in media content` | Wrong model for edit_image | Use `capability-001` or `ingredients-preview` |
| `PERMISSION_DENIED: cloud-lvm@prod.google.com` | GCS bucket not authorized | Add `user:cloud-lvm@prod.google.com` as `objectViewer` |
| `Cannot process more than 2 reference images for non-square` | API limit | Auto-truncated: 1:1→3 refs, others→2 refs |
| `Allow (Adults only)` blocked virtual-try-on | `ALLOW_ADULT` filter too strict | Fixed: uses `ALLOW_ALL` |
| `SSL UNEXPECTED_EOF` | Network interruption | Fixed: auto-retries 3× with backoff |
| `404 product-recontext unavailable` | Requires allowlist approval | Apply at Google Cloud Console |
| `Multiple candidates not enabled` | Gemini image model limitation | Fixed: loops N separate requests |
| `Unknown mime type` | google-genai 1.63+ API change | Fixed: uses `UploadFileConfig` |
| `ModuleNotFoundError: google.genai` | Wrong Python env | `pip install` into ComfyUI's `.venv` |

---

## 🔧 Architecture

```
NanoBananaImagen.generate()
│
├── make_image(tensor)
│   ├── GCS mode  → tensor → PNG → GCS (cached permanently by MD5 hash)
│   └── bytes mode → tensor → PNG bytes (sent each request)
│
├── Model routing
│   ├── edit_image path
│   │   ├── imagen-3.0-capability-001  → SubjectReferenceImage(PERSON)
│   │   └── imagen-4.0-ingredients-preview → ContentReferenceImage
│   └── recontext_image path
│       ├── virtual-try-on-001          → person_image + ProductImage(clothing)
│       └── imagen-product-recontext    → ProductImage(1-3) + prompt
│
└── Returns batch tensor (B, H, W, C) — native ComfyUI multi-image format
```

---

## 📋 Requirements

```
google-genai>=1.63.0
google-cloud-storage>=2.0.0
Pillow
torch
numpy
```

---

## 📄 License

MIT — feel free to use, modify, and distribute.
