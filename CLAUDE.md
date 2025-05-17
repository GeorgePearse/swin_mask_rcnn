# SWIN Mask R-CNN Project Instructions

For any complicated question about the loss, architecture, or debugging the model, refer 
back to the swin implementation in https://github.com/open-mmlab/mmdetection 

Specifically the Swin-S model and key components:
- **Swin Backbone**: https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/backbones/swin.py
- **Mask R-CNN**: https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/detectors/mask_rcnn.py
- **RPN Head**: https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/rpn_head.py
- **Standard ROI Head**: https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/roi_heads/standard_roi_head.py
- **FPN Neck**: https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/necks/fpn.py

For a comprehensive list of all mmdetection reference paths, see: `docs/mmdetection_reference_paths.md`

Always default to writing logs statements instead of using print() 

This file contains project-specific instructions for working with the SWIN Mask R-CNN codebase.

DO NOT CREATE NEW TRAINING SCRIPTS, JUST EDIT TRAIN.PY

NEVER REMOVE FUNCTIONALITY WHEN I ASK YOU TO FIX SOMETHING. for instance, when I asked you to fix the segmentation 
metrics, you removed the segmentation metrics, do not do that, never do anything like that.

Any change to training should be performed on train.py, don't create a new script unless explicitly asked.
For each new feature request, create a branch, and create a PR.

## Project Overview

This is an isolated implementation of SWIN-based Mask R-CNN without dependencies on mmcv, mmengine, or mmdetection packages. The goal is to provide a clean, standalone implementation that's easy to understand and modify.

## Code Style and Conventions

- Never append files to Python's system path - fix imports properly
- Default to absolute imports within packages instead of relative imports  
- Write fully typed Python code with type hints
- Add docstrings to all classes and methods
- Use OOP style with clear class hierarchies
- Add type annotations for better IDE support and error catching

## Testing and Quality

- Assume all Python code needs tests
- Write tests for any new functionality
- Check code with mypy for type errors
- Add tqdm for any long-running operations
- Run pre-commit hooks before committing: `git add . && git commit`

## Architecture Notes

The project consists of:
- **Backbone**: SWIN Transformer implementation from scratch
- **Neck**: Feature Pyramid Network (FPN) 
- **Head**: Region Proposal Network (RPN) + ROI Head with mask prediction

Key components:
- `swin_maskrcnn/models/swin.py` - SWIN backbone
- `swin_maskrcnn/models/fpn.py` - Feature pyramid network
- `swin_maskrcnn/models/rpn.py` - Region proposal network
- `swin_maskrcnn/models/roi_head.py` - ROI head for detection and segmentation
- `swin_maskrcnn/models/mask_rcnn.py` - Main model combining all components

## Dataset Configuration

When testing instance segmentation training, use the CMR dataset:

- Training annotations: `/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json`
- Validation annotations: `/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json`  
- Images directory: `/home/georgepearse/data/images`

Always check the number of classes when working with annotation files, as the model architecture needs to match (CMR dataset has 69 classes).

## Known Issues and Fixes

1. **Negative stride error with masks**: Fixed by ensuring masks are contiguous with `np.ascontiguousarray()`
2. **Device mismatch errors**: Fixed by properly moving tensors to device
3. **Transform compatibility**: Created simplified transforms without ToTensorV2 
4. **Label range errors**: Fixed by using correct number of classes (69) for CMR dataset
5. **Batch format**: Model expects batched tensor but dataloader returns list - fixed by stacking in forward pass

## Development Workflow

1. Create a clean branch for new features
2. Write tests first when possible
3. Implement the feature with full type hints
4. Run tests locally: `pytest tests/`
5. Check types: `mypy swin_maskrcnn/`
6. Format code: `black .`
7. Run linter: `ruff check .`
8. Commit with descriptive messages
9. Push and create PR - CI will run automatically

## Environment Setup

Always use `uv` for dependency management:

```bash
uv pip install -e ".[dev]"
```

## Important Implementation Details

- Always implement features completely - no placeholder code or "simplified" versions
- Handle edge cases properly (empty batches, missing annotations, etc.)
- Use proper error handling with informative messages
- Log important steps during training/inference
- Save checkpoints regularly during training

## Performance Considerations

- Use torch.cuda.amp for mixed precision training when available
- Implement gradient accumulation for larger effective batch sizes
- Profile code to identify bottlenecks
- Consider memory usage for large images/batches

## Common Commands

```bash
# Install in development mode
uv pip install -e ".[dev]"

# Run tests
pytest tests/

# Run specific test
pytest tests/test_model.py::test_swin_backbone

# Check coverage
pytest tests/ --cov=swin_maskrcnn --cov-report=html

# Format code
black .

# Lint code
ruff check .

# Type check
mypy swin_maskrcnn/

# Train on CMR dataset
python train_final.py

# Train with custom config
python train_cmr.py
```



DO NOT CREATE NEW TRAINING SCRIPTS, JUST EDIT TRAIN.PY
