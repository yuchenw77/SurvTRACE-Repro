# SurvTRACE Repro

This is a repro for the paper "SurvTRACE: Transformers for Survival Analysis with Competing Events”

### Reference to Original Paper

Wang, Zifeng, and Jimeng Sun. "SurvTRACE: Transformers for Survival Analysis with Competing Events." arXiv preprint arXiv:2110.00855 (2021).

original repo: [https://github.com/RyanWangZf/SurvTRACE](https://github.com/RyanWangZf/SurvTRACE)

### Dependencies

see requirements.txt

### Data Download Instruction

1. For metabric:

```
from pycox.datasets import metabric
df = metabric.read_df()
```

1. For support:

```
from pycox.datasets import support
df = support.read_df()
```

1. For seer:

Follow the instruction on [https://github.com/RyanWangZf/SurvTRACE](https://github.com/RyanWangZf/SurvTRACE)

### Command for Preprocessing, training, evaluating

```r
python repro_metabric.py
python repro_support.py
python repro_seer.py
```

### Results

See /Result