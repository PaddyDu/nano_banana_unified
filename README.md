# Nano Banana Unified - ComfyUI Node

A unified ComfyUI custom node for Google's Gemini Image Generation models. This node integrates both **Gemini 2.5 Flash** and **Gemini 3 Pro** into a single, powerful interface, supporting advanced operations like generation, editing, style transfer, and object insertion.

![Banner](https://github.com/user-attachments/assets/placeholder)<!-- You might want to add a screenshot here later -->

## Features

*   **Unified Model Support**: Seamlessly switch between `gemini-2.5-flash-image` and `gemini-3-pro-image-preview`.
*   **Multiple Operations**:
    *   `generate`: Text-to-Image generation.
    *   `edit`: Modify existing images using prompts.
    *   `styletransfer`: Apply style from reference images.
    *   `objectinsertion`: Insert objects into scenes.
*   **Files API Integration**: Automatically handles uploading reference images to Google's Files API and manages caching (TTL default 48h) to save bandwidth and upload time.
*   **Multi-Reference Support**: Use up to 5 reference images for complex conditioning.
*   **Advanced Control**:
    *   Parameters for `seed`, `top_p`, `top_k`, `candidate_count`, and `guidance_scale`.
    *   Aspect ratio selection.
    *   Safety filter settings.
*   **Streaming Logs**: Real-time feedback in the console/UI for long-running generation tasks (Gemini 3).

## Installation

1.  Navigate to your ComfyUI custom nodes directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/PaddyDu/nano_banana_unified.git
    ```
3.  Install the required Python dependency:
    ```bash
    pip install google-genai
    ```
    *(Note: Ensure you are using the python environment associated with your ComfyUI installation)*

## Configuration

You need a Google Gemini API Key to use this node. You can get one from [Google AI Studio](https://aistudio.google.com/).

### Method 1: `config.json` (Recommended)
1.  Locate `config.example.json` in the node folder.
2.  Rename it to `config.json`.
3.  Edit the file and paste your API key:
    ```json
    {
        "GEMINIAPIKEY": "YOUR_ACTUAL_API_KEY_HERE",
        "filecache": {}
    }
    ```

### Method 2: Environment Variable
Set the `GEMINIAPIKEY` environment variable on your system before launching ComfyUI.

### Method 3: Node Input
You can optionally pass the API key directly into the `apikey` widget string on the node itself.

## Usage

1.  **Add Node**: Right-click in ComfyUI -> `NanoBananaUnified` (or search for it).
2.  **Select Model**: Choose between `gemini-2.5-flash-image` (faster) or `gemini-3-pro-image-preview` (higher quality/new features).
3.  **Select Operation**: Choose what you want to do (e.g., `generate`, `edit`).
4.  **Prompt**: Enter your text prompt.
5.  **References (Optional)**: Connect `IMAGE` inputs if you are doing editing or style transfer.

## Caching Behavior
The node automatically caches uploaded reference images in `config.json` under `filecache`. It checks if a file with the same signature has already been uploaded and if the URI is still valid (within 48 hours). This speeds up subsequent workflow runs using the same images.

## License
MIT
