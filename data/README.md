# Data

The test images are not stored in this repository. They are published on Zenodo:

- **R2Net custom dataset** — 18 grayscale RAW images, from 416×240 to 2560×2048:
  <https://zenodo.org/records/20268850> (doi: 10.5281/zenodo.20268850)
- **Benchmark results** (timings and hardware telemetry, float32 vs. float64):
  <https://zenodo.org/records/20347920> (doi: 10.5281/zenodo.20347920)

Download the images and place the `.raw` files in this folder. The notebook expects, for example:

```
data/n1_GRAY.ube8_1_2560_2048.raw
```

## File name convention

`<name>.<encoding>_<channels>_<height>_<width>.raw`

| Field | Example | Meaning |
| --- | --- | --- |
| encoding | `ube8` | unsigned, big-endian, 8 bits per sample (`ube16`, `sbe16`, `ule16`… are also supported) |
| channels | `1` | grayscale |
| height × width | `2560_2048` | image size in pixels |

RAW files have no header, so these values must be passed to `load_raw_image` (see `src/utils/io.py`).
