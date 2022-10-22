# GDS-Conv

## Enviroment

1. Python == 3.6
2. torch == 1.8, torchaudio == 0.8.0, cuda == 10.2
3. Fairseq-S2T == 0.12.2

## Code Structure

```markdown
|
|—— gmm.py
|—— s2t_transformer.py
|—— subsampling.py
|—— utils.py
├── st
│   ├── base.yaml
|   └── dsconv.sh
└── asr
    ├── base.yaml
    └── dsconv.yaml
```

## Usage

To use GDS-Conv, Following the dsconv config:

```yaml
subsampling-type: gmm
# gmm: GDS-Conv
# thresh :TDS-Conv

shrink-value: -0.2
# TDS-Conv's thresh value

random-mask: False
# randomly set 10% percent of IM to false

pool-thresh: 0.0

gmm-iter: 20
# gmm max iter num

gmm-iter-per: 10
# each bath of input's iter num

gmm-prob: 30
# gmm min allow -prob
# 30(ASR,gim-dim=40)
# 70(ASR,gim-dim=80)
# 80(ST,gim-dim=80)

gmm-num: 2
# gmm distribution num

gmm-dim: 80
# uses filter bank's first 80 diensions to fit gmm

```



### ASR model description



### ST model description



### GDS-Conv

step1: select 2000 sentences from train set

step2: start your training with Fairseq on single GPU with step1's sub-train set, and stop and save after 10 or 20 steps to fit GMM

step3: restart your training with 8 GPUs or more with the step2's checkpoint and complete train set

step4: inference with step3's checkpoint



## ST Score

| Model      | En-De BLEU  ↑ | En-De Reduction Rate ↓ | En-Fr BLEU ↑ | En-Dr Reduction Rate ↓ |
| ---------- | ------------- | ---------------------- | ------------ | ---------------------- |
| Fairseq    | 22.70         | 25 (%)                 | 32.90        | 25                     |
| NeurST     | 22.80         | 25                     | 33.30        | 25                     |
| Baseline4x | 22.57         | 25                     | 33.01        | 25                     |
| Baseline8x | 21.98         | **13**                 | 31.72        | **13**                 |
| TDS-Conv   | 22.55         | 22                     | 33.00        | 22                     |
| GDS-Conv   | **23.11**     | 22                     | **33.40**    | 22                     |

hint: TDS-Conv usesenergy thresh to raplace GMM.

## ASR Score

| Model      | dev-clean | dev-other | test-clean | test-other | R    |
| ---------- | --------- | --------- | ---------- | ---------- | ---- |
| WeNet      | -         | -         |            |            |      |
| Baseline4x | 3.04      |           |            |            |      |
| Baseline8x | 2.97      |           |            |            |      |
| TDS-Conv   | 2.97      |           |            |            |      |
| GDS-Conv   |           |           |            |            |      |

