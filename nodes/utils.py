"""Shared utilities: config I/O, constants."""

import os
import json

# Root directory of the plugin
PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CONFIG_FILENAME = "config.json"
CACHE_TTL_SECONDS = 48 * 3600  # Gemini Files API URI TTL


def get_config() -> dict:
    try:
        config_path = os.path.join(PLUGIN_DIR, CONFIG_FILENAME)
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(config: dict):
    try:
        config_path = os.path.join(PLUGIN_DIR, CONFIG_FILENAME)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    except Exception:
        pass
