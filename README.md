# Cross-Sensor Optical Ship Detection with Mixed Fine-Tuning

Domain-adaptive YOLO26-S ship detection from **LEVIR-Ship (GF-1/GF-6)** to **Sentinel-2**, with source-domain retention analysis and direct inference on unseen Sentinel-2 imagery.

## Overview

This project investigates whether a ship detector trained on one optical satellite domain can be adapted to **Sentinel-2** while minimizing loss of its original detection capability.

A YOLO26-S detector was first trained on **LEVIR-Ship**, constructed from **GaoFen-1 (GF-1) and GaoFen-6 (GF-6)** optical imagery. The source model achieved strong performance on the independent LEVIR test set. Mixed fine-tuning was then performed using LEVIR and Sentinel-2 ship imagery at controlled sampling ratios.

The goal is to balance two competing objectives:

1. **Target-domain adaptation** — expose the detector to Sentinel-2 imagery.
2. **Source-domain retention** — minimize degradation of the original LEVIR detection capability.

Among the tested conditions summarized here, **LEVIR : Sentinel-2 = 4 : 1** provided the strongest overall retention/adaptation trade-off.

## Highlights

- YOLO26-S optical satellite ship detection
- Cross-sensor adaptation: **GF-1/GF-6 → Sentinel-2**
- Mixed fine-tuning with source-domain replay
- Multiple source : target sampling-ratio experiments
- Independent LEVIR test evaluation
- Source-domain retention analysis
- Direct inference on unseen Busan Sentinel-2 imagery
- Small/tiny-ship detection under medium-resolution satellite imagery

## Motivation

Public Sentinel-2 datasets specifically designed for **ship object detection with bounding-box annotations** are relatively limited. LEVIR-Ship provides a larger source dataset for learning ship-detection representations, while Sentinel-2 is the target sensor domain.

Rather than training only a target-specific detector, this project studies a deployment-oriented question:

> **How can an existing detector be extended to a new satellite sensor while retaining as much previously learned detection capability as possible?**

This makes the project an **adaptation–retention trade-off** problem rather than a simple benchmark-maximization task.

## Cross-Sensor Setting

| Property | Source Domain | Target Domain |
|---|---|---|
| Dataset / imagery | LEVIR-Ship | Sentinel-2 ship imagery |
| Satellite | GF-1 / GF-6 | Sentinel-2 |
| Sensor type | Multispectral optical | Multispectral optical |
| Channels used | RGB | RGB (B4/B3/B2) |
| Spatial resolution | ~16 m | 10 m for RGB |
| Target | Tiny / small ships | Small ships |
| Role | Source-domain learning | Target-domain adaptation |

Although both domains are multispectral optical satellite imagery, their image distributions differ because they come from different sensors, resolutions, acquisition conditions, preprocessing pipelines, backgrounds, and target appearances.

## Datasets

### 1. LEVIR-Ship

LEVIR-Ship is the primary **source-domain** dataset.

```text
Original scenes : 85
Image patches   : 3,896
Ship instances  : 3,219
Patch size      : 512 × 512
Resolution      : ~16 m
Classes         : 1 (Ship)
```

The independent LEVIR test set is retained to quantify how much source-domain capability remains after Sentinel-2 adaptation.

### 2. Sentinel-2 Ship Detection Dataset

Sentinel-2 ship imagery is used as the **target-domain adaptation dataset**.

RGB bands:

```text
Red   : B4
Green : B3
Blue  : B2
Resolution: 10 m / pixel
```

The target data are mixed with LEVIR training samples during fine-tuning rather than completely replacing the source data.

### 3. Unseen Busan Sentinel-2 Imagery

The trained models are also applied to Sentinel-2 imagery over the Busan coastal/port region.

Because independent ship bounding-box ground truth is not available for this scene, the result is treated as **qualitative deployment-domain evidence**, not quantitative accuracy.

## Sentinel-2 RGB Preprocessing

Sentinel-2 reflectance bands are converted to 8-bit RGB before YOLO inference.

```python
def reflectance_to_uint8(x, vmin=0.02, vmax=0.25):
    x = x.astype(np.float32)
    x = (x - vmin) / (vmax - vmin)
    x = np.clip(x, 0, 1)
    x = np.power(x, 1 / 1.4)
    return (x * 255).astype(np.uint8)

rgb = np.stack([
    reflectance_to_uint8(red),
    reflectance_to_uint8(green),
    reflectance_to_uint8(blue)
], axis=-1)
```

## Baseline: LEVIR-Only Training

The YOLO26-S source model was evaluated on the independent LEVIR test set.

| Metric | LEVIR-Only |
|---|---:|
| Precision | **0.8703** |
| Recall | **0.8678** |
| mAP@0.5 | **0.8622** |
| mAP@0.5:0.95 | **0.3172** |

These values define the source-domain reference before adaptation.

## Mixed Fine-Tuning Strategy

Instead of fine-tuning exclusively on Sentinel-2, source and target samples are mixed:

```text
LEVIR knowledge
      +
Sentinel-2 exposure
      ↓
Cross-sensor adaptation
      ↓
Reduced source-domain forgetting
```

Multiple LEVIR : Sentinel-2 ratios were tested. Each ratio should be initialized from the **same LEVIR-only checkpoint** for a fair comparison.

The purpose of source replay is not merely to maximize LEVIR performance, but to reduce source-domain degradation while the detector is exposed to the new Sentinel-2 domain.

## Mixed Fine-Tuning Results

Mixed models were fine-tuned for **30 epochs**.

### Independent LEVIR Test

| Training Strategy | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| **LEVIR-only baseline** | **0.8703** | **0.8678** | **0.8622** | **0.3172** |
| Mixed FT 2:1 | 0.7578 | 0.6938 | 0.7730 | 0.2908 |
| **Mixed FT 4:1 (best.pt)** | **0.7958** | **0.7717** | **0.7986** | **0.2749** |
| Mixed FT 5:1 (best.pt) | 0.7817 | 0.7319 | 0.7650 | 0.2575 |
| Mixed FT 5:1 (last.pt) | 0.7750 | 0.7717 | 0.7829 | 0.2685 |

Additional ratios can be added when their final independent test results are available.

## Why 4:1?

Compared with the LEVIR-only baseline, the 4:1 model changes as follows:

```text
Precision    : 0.8703 → 0.7958   (-0.0745)
Recall       : 0.8678 → 0.7717   (-0.0961)
mAP@0.5      : 0.8622 → 0.7986   (-0.0636)
mAP@0.5:0.95 : 0.3172 → 0.2749   (-0.0423)
```

Relative source-domain mAP@0.5 retention:

```text
0.7986 / 0.8622 ≈ 92.6%
```

Thus, the 4:1 model retains approximately **92.6% of the original LEVIR mAP@0.5** after incorporating Sentinel-2 training data.

Increasing the source ratio further did not monotonically improve retention:

```text
4:1 best mAP@0.5 = 0.7986
5:1 best mAP@0.5 = 0.7650
```

This suggests that simply increasing source-domain sampling does not guarantee a better adapted model.

## Source-Domain Retention

The adaptation result should **not** be described as preserving LEVIR performance without loss. There is measurable degradation.

A more precise interpretation is:

> **The detector was adapted toward Sentinel-2 while retaining 92.6% of the original LEVIR mAP@0.5 under the selected 4:1 mixed fine-tuning condition.**

This explicitly treats adaptation as a trade-off between new-domain exposure and previously learned source-domain capability.

## Small-Object Localization

LEVIR-Ship and Sentinel-2 ship detection both involve small/tiny targets. For a small bounding box, a displacement of only a few pixels can cause a large IoU change.

This is a plausible contributor to the gap between:

```text
mAP@0.5      = 0.7986
mAP@0.5:0.95 = 0.2749
```

However, this should be treated as a hypothesis until AP75, false-positive cases, and prediction/ground-truth box alignment are analyzed.

## Sentinel-2 Deployment Inference

The final mixed model is applied to an unseen Busan Sentinel-2 scene.

Recommended figure:

```markdown
![Busan Sentinel-2 Result](figures/LEVIR_S2_4to1_Busan_result.png)
```

For a fair qualitative comparison, use the same:

- Sentinel-2 scene
- RGB preprocessing
- confidence threshold
- input size
- NMS settings

for both the **LEVIR-only** and **Mixed FT 4:1** models.

Higher and more stable confidence on visually apparent ships can be presented as **qualitative evidence of stronger target-domain response**, but not as higher precision/recall without ground truth.

## Experimental Interpretation

```text
Strong source detector
        ↓
New satellite sensor
        ↓
Domain gap
        ↓
Target adaptation
        ↓
Source-domain degradation
        ↓
Mixed-domain replay
        ↓
Adaptation–retention trade-off
```

The main engineering question is:

> **How much source-domain performance is lost during adaptation to a new sensor, and how can that loss be minimized?**

## Key Findings

1. **The LEVIR-only detector provides a strong source baseline.**  
   P = 0.8703, R = 0.8678, mAP@0.5 = 0.8622.

2. **Cross-sensor adaptation causes measurable source-domain degradation.**

3. **Mixed fine-tuning provides a mechanism for controlling the adaptation–retention trade-off.**

4. **The reported 4:1 condition provides the strongest overall source-retention result among the summarized mixed experiments.**  
   LEVIR mAP@0.5 = 0.7986 after adaptation.

5. **Approximately 92.6% of the original LEVIR mAP@0.5 is retained.**

6. **More source data does not automatically improve retention.**  
   The 5:1 experiment performed worse than 4:1 on the independent LEVIR test.

7. **Strict localization remains challenging for small ships.**  
   Further AP75 and error analysis is needed.

## Important Evaluation Note

LEVIR test results are **quantitative** because ground-truth bounding boxes are available.

The Busan Sentinel-2 result is currently **qualitative** because independent verified ship bounding boxes are not available for that scene.

Therefore:

- LEVIR Precision / Recall / mAP are quantitative metrics.
- Sentinel-2 boxes and confidence scores are model outputs.
- Higher target-domain confidence alone does not prove higher accuracy.
- The **92.6%** figure refers specifically to **LEVIR source-domain mAP@0.5 retention**.
- Rigorous target-domain accuracy requires an independent annotated Sentinel-2 validation/test set.

## Repository Structure

```text
Optical_Ship_Detection_Domain_Adaptation/
│
├── notebooks/
│   ├── Train_YOLO_LEVIR.ipynb
│   ├── Mixed_Finetune_LEVIR_Sentinel2.ipynb
│   ├── Test_LEVIR.ipynb
│   └── Sentinel2_Inference.ipynb
│
├── figures/
│   ├── Workflow.png
│   ├── LEVIR_only_Busan_S2_result.png
│   ├── LEVIR_S2_2to1_Busan_result.png
│   ├── LEVIR_S2_4to1_Busan_result.png
│   └── Ratio_Performance_Comparison.png
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

Adjust notebook and figure names to match the repository.

## Suggested Summary Figure

```text
               LEVIR-Ship
                GF-1/GF-6
                    │
                    ▼
             YOLO26-S Baseline
                    │
       ┌────────────┴────────────┐
       ▼                         ▼
  LEVIR Test                Sentinel-2
mAP50 = 0.8622              Domain Gap
                                 │
                                 ▼
                      Mixed Fine-Tuning
                      LEVIR : S2 = 4 : 1
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
              LEVIR Test                Busan Sentinel-2
            mAP50 = 0.7986             Qualitative Inference
            92.6% retained
```

## Requirements

Experiments were developed primarily in Google Colab.

```text
Python
PyTorch
TorchVision
Ultralytics
OpenCV
NumPy
Matplotlib
PyYAML
h5py
```

Example installation:

```bash
pip install ultralytics opencv-python numpy matplotlib pyyaml h5py
```

## Data Availability

Large satellite datasets and imagery are not included directly in this repository. Users should obtain datasets from their original providers and follow the corresponding licenses and terms of use.

- **LEVIR-Ship** — source-domain optical ship-detection dataset
- **Sentinel-2 Ship Detection Dataset** — target-domain adaptation dataset
- **Busan Sentinel-2 imagery** — qualitative deployment-domain inference

Large trained checkpoints may need to be stored separately.

## Limitations

### 1. Limited Quantitative Target-Domain Evaluation

The Busan deployment scene does not currently have independent verified ship bounding-box annotations. Target-domain deployment results are therefore interpreted qualitatively.

### 2. Source-Domain Performance Loss

The 4:1 model retains approximately 92.6% of baseline LEVIR mAP@0.5, but the original source-domain performance is not completely preserved.

### 3. Small-Object Localization

mAP@0.5:0.95 remains substantially below mAP@0.5. Further analysis is required to separate tiny-object localization difficulty from annotation uncertainty and other detector errors.

### 4. Ratio Selection

The 4:1 ratio is the strongest reported condition among the experiments summarized here. It should not be interpreted as universally optimal.

### 5. Cross-Sensor Generalization

More Sentinel-2 regions, dates, sea states, coastal environments, and independently annotated scenes are required for a comprehensive generalization study.

## Future Work

- Quantitative evaluation on an independent Sentinel-2 test set
- Manual annotation or AIS-assisted validation of Busan imagery
- False-positive and false-negative analysis
- AP50 / AP75 comparison
- Hard-negative mining for ports and coastal structures
- Multi-scale training for small ships
- Feature-level domain alignment
- Domain-adversarial adaptation
- Evaluation on additional Sentinel-2 regions and acquisition dates

## Conclusion

This project demonstrates a practical **cross-sensor domain adaptation problem in optical satellite ship detection**.

The LEVIR-only YOLO26-S baseline achieved:

```text
Precision    : 0.8703
Recall       : 0.8678
mAP@0.5      : 0.8622
mAP@0.5:0.95 : 0.3172
```

After mixed fine-tuning with Sentinel-2 data using a **4:1 LEVIR-to-Sentinel-2 ratio**:

```text
Precision    : 0.7958
Recall       : 0.7717
mAP@0.5      : 0.7986
mAP@0.5:0.95 : 0.2749
```

The adapted detector therefore retains approximately **92.6% of the original LEVIR mAP@0.5** while being exposed to the Sentinel-2 target domain.

The main result is not adaptation without cost, but that **mixed-domain fine-tuning provides a practical way to control the trade-off between adaptation to a new optical satellite sensor and retention of previously learned source-domain detection capability**.

## Author

Jinho Lee  
Satellite Oceanography Laboratory  
Seoul National University
