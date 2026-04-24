# Repository Guidelines

## Project Structure & Module Organization
`app.py` hosts the Gradio interface, while `inference.py` is the batch CLI entry point for text-to-SVG and image-to-SVG generation. Core runtime helpers live in `decoder.py` and `tokenizer.py`, and shared settings are centralized in `config.yaml`. The `deepsvg/` package contains SVG parsing, model, utility, and legacy GUI code. Use `metrics/` for evaluation scripts, `data/` for conversion helpers and small sample inputs, `assets/` plus `examples/` for demo media and prompts, `scripts/windows/` for local runtime helpers, and `third_party/cairo/` for the pinned Cairo source submodule.

## Build, Test, and Development Commands
`git submodule update --init --recursive` fetches pinned third-party sources after clone.  
`uv python pin 3.12 && uv sync` creates the supported Python 3.12 environment.  
`uv run python app.py` launches the local Gradio demo; on Windows, prefix it with `pwsh -File scripts/windows/with_cairo_runtime.ps1` if Cairo is not already on `PATH`.  
`uv run python inference.py --task text-to-svg --input prompts.txt --output ./output_text --save-all-candidates` runs batch text generation.  
`uv run python inference.py --task image-to-svg --input ./examples --output ./output_image --save-all-candidates` runs image-conditioned generation.  
`uv run huggingface-cli download OmniSVG/OmniSVG1.1_8B --local-dir /PATH/TO/OmniSVG1.1_8B` downloads model weights inside the managed environment.  
When overriding model paths, keep the official pairings aligned: `4B` uses `Qwen/Qwen2.5-VL-3B-Instruct` with `OmniSVG/OmniSVG1.1_4B`, and `8B` uses `Qwen/Qwen2.5-VL-7B-Instruct` with `OmniSVG/OmniSVG1.1_8B`.  
Use `uv run python metrics/compute_fid.py ...` or the related `metrics/compute_*.py` scripts to evaluate generated outputs; several metric scripts need extra research dependencies beyond the core `uv sync`.

## Coding Style & Naming Conventions
Match the existing Python style: 4-space indentation, snake_case for functions and variables, PascalCase for classes, and UPPER_CASE for module-level constants. Prefer small helper functions over inline logic in `app.py` and `inference.py`. Keep tunable behavior in `config.yaml` instead of hard-coding paths or thresholds. There is no committed formatter or linter config, so keep imports grouped and follow PEP 8 manually.

## Testing Guidelines
This repository does not currently include a dedicated `tests/` directory. Validate changes with focused smoke tests: launch `uv run python app.py` for UI changes, run a targeted `uv run python inference.py ...` command for generation changes, and use the relevant `uv run python metrics/compute_*.py` script for evaluation work. If you add automated tests, place them under a new `tests/` tree and use `test_<module>.py` naming.

## Commit & Pull Request Guidelines
Recent history uses short, direct subjects such as `Update README.md` and `OmniLottie release news`. Keep commit messages brief, imperative, and limited to one logical change. Pull requests should summarize behavior changes, list validation commands, link related issues or dataset/model references, and include screenshots for `app.py` UI edits or sample SVG/PNG outputs for inference changes.

## Configuration Tips
Do not commit downloaded model weights, Hugging Face cache data, `.venv/`, `.cairo-runtime/`, `.cairo-runtime64/`, or local output directories such as `output_text/` and `output_image/`. Keep machine-specific paths out of tracked files and document any new external system dependency, especially CUDA or Cairo requirements, in `README.md`. On Windows, the supported GPU target is the PyTorch CUDA 13.0 wheel set configured in `pyproject.toml`, and `CairoSVG` still requires a native **64-bit** Cairo runtime with `libcairo-2.dll` on `PATH`. Treat `third_party/cairo/` as an upstream submodule: update it by changing the pinned commit, not by copying files into the main tree.
