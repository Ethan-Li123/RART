# Regional Anchor Routing for Multi-Object Motion Transfer

Yuze Li<sup>1</sup>, Fuchen Long<sup>2</sup>, Xinlong Sun<sup>2</sup>, Cheng Yan<sup>1§</sup>, Xinyu Zhang<sup>3</sup>, Xi Chen<sup>2§</sup>

<sup>1</sup> Tianjin University  
<sup>2</sup> Smart Creation Platform Department, Online Video BU, Tencent  
<sup>3</sup> University of Auckland

Work done during the Tencent internship.  
§ Corresponding Authors

RART transfers independent reference motions to designated objects in a target image, using Wan2.2-TI2V-5B and a DisMo motion encoder.

[Paper](https://arxiv.org/abs/2610.39478) · [Project page](https://ethan-li123.github.io/RART_page/) · [RART weights](https://huggingface.co/llyyzzz/RART)

## Environment

Run these commands from the project root:

```bash
conda create -n rart python==3.10
conda activate rart
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
pip install flash-attn==2.8.3+pt2.8.0cu129 --extra-index-url https://miropsota.github.io/torch_packages_builder
pip install -r requirements.txt
```

## Model weights

Download the native [Wan2.2-TI2V-5B checkpoint](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B):

```bash
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir weights/Wan2.2-TI2V-5B
```

Download the [DisMo motion extractor](https://huggingface.co/CompVis/DisMo):

```bash
hf download CompVis/DisMo motion_extractor_large.pt --local-dir weights
```

Download the ViT-L/14 backbone **with registers** from [DINOv2](https://github.com/facebookresearch/dinov2), saving it under the filename used by RART:

```bash
wget -O weights/dinov2_vitl14_reg.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth
```

The RART checkpoint is available at [llyyzzz/RART](https://huggingface.co/llyyzzz/RART). Download it with:

```bash
hf download llyyzzz/RART temporal_motion.pt --local-dir outputs/checkpoint-200000
```

The resulting project-relative paths are:

```text
weights/
  Wan2.2-TI2V-5B/             # Wan model shards, T5, tokenizer and VAE
  motion_extractor_large.pt  # DisMo
  dinov2_vitl14_reg.pth       # DINOv2
outputs/
  checkpoint-200000/
    temporal_motion.pt       # RART
```

If weights are already available locally, place them at the paths above or edit the path arguments in `scripts/run_inference.sh`.

## Test data

Five examples are included in `data/RART_test_data/`:

```text
data/RART_test_data/
  inference.csv
  README.md
  case_01/  # two people
  case_02/  # creature and owl
  case_03/  # capybara and person
  case_04/  # elephant and tiger
  case_05/  # cartoon dog and koala
```

Each example contains two reference videos, their foreground masks, a target image and two target-object masks. Reference, mask, person and action entries follow the same object order.

## Inference

Activate the environment and run:

```bash
conda activate rart
bash scripts/run_inference.sh
```

For one GPU, set `CUDA_VISIBLE_DEVICES=0`, change `--num_processes` to 1 and remove `--multi_gpu` in the script.

Run only the first example, or select a subset:

```bash
bash scripts/run_inference.sh --csv_limit 1
bash scripts/run_inference.sh --csv_start 1 --csv_limit 2
```

Outputs are saved as `outputs/inference/case_01.mp4` through `case_05.mp4`. Add `--concat_ref` to include the reference videos beside the generated result:

```bash
bash scripts/run_inference.sh --concat_ref
```

Sampling defaults are defined in `scripts/inference_temporal.py`: 49 frames, 480 x 736 resolution, 50 steps, text guidance 3, motion guidance 5 and seed 42.

T5 CPU offloading precomputes text embeddings before loading Wan, then moves T5 to CPU. It is enabled by default; pass `--no_t5_cpu_offload` to disable it. VAE offloading is also enabled; use `--no_vae_offload` to disable it.

## Source layout

```text
scripts/          # inference entry point and launcher
wan/              # Wan backbone, text encoder, VAE and solvers
wan_motion_attn/  # motion encoder and regional attention
DisMo/            # bundled motion-extractor source
```

## Citation

If you find RART useful, please cite:

```bibtex
@misc{li2026regionalanchorrouting,
  title={Regional Anchor Routing for Multi-Object Motion Transfer},
  author={Yuze Li and Fuchen Long and Xinlong Sun and Cheng Yan and Xinyu Zhang and Xi Chen},
  year={2026},
  eprint={2610.39478},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.39478}
}
```

Acknowledgements: [Wan2.2](https://github.com/Wan-Video/Wan2.2) and [DisMo](https://github.com/CompVis/DisMo).
