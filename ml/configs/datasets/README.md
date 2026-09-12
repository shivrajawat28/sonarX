# Dataset definitions

One YAML per dataset (created in Phase 3 / Step 5 by `inspect_dataset.py` findings).
These configs are where candidate classes are declared from dataset evidence —
the closing point of OPEN decision #1. Example shape:

```yaml
name: sonar-v0.1
root: datasets/processed/sonar-v0.1
format: yolo          # yolo | coco | voc | csv (converter registry keys)
candidate_classes:    # verbatim from dataset review, never invented
  - shipwreck
  - pipe
split_seed: 42
split_fractions: { train: 0.7, val: 0.2, test: 0.1 }
notes: reviewed in docs/dataset-notes/<file>.md
```

No application code may hardcode these names; the model `class_map` is built from
training runs that consume this config.
