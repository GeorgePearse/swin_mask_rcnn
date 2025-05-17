"""Create a checkpoint with correctly initialized biases."""
import torch
from pathlib import Path
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from swin_maskrcnn.utils.load_coco_weights import load_coco_weights
from swin_maskrcnn.utils.logging import setup_logger


def main():
    logger = setup_logger()
    
    # Initialize model with improved bias initialization
    logger.info("Creating model with correct bias initialization...")
    model = SwinMaskRCNN(num_classes=69)
    
    # Load COCO pretrained weights
    logger.info("Loading COCO pretrained weights...")
    missing_keys, unexpected_keys = load_coco_weights(model, num_classes=69)
    logger.info(f"Missing keys: {len(missing_keys)}")
    logger.info(f"Unexpected keys: {len(unexpected_keys)}")
    
    # Verify biases
    logger.info("\nChecking biases after loading:")
    cls_bias = model.roi_head.bbox_head.fc_cls.bias.detach().cpu().numpy()
    logger.info(f"Classification bias shape: {cls_bias.shape}")
    logger.info(f"Background bias: {cls_bias[0]:.4f}")
    logger.info(f"Foreground biases - mean: {cls_bias[1:].mean():.4f}, "
               f"min: {cls_bias[1:].min():.4f}, max: {cls_bias[1:].max():.4f}")
    
    # Create checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'num_classes': 69,
        'config': {
            'init_type': 'coco_pretrained_with_correct_biases'
        }
    }
    
    # Save checkpoint
    save_path = Path("/home/georgepearse/swin_maskrcnn/checkpoints/coco_initialized_corrected_biases.pth")
    save_path.parent.mkdir(exist_ok=True)
    torch.save(checkpoint, save_path)
    logger.info(f"\nSaved checkpoint to: {save_path}")
    
    # Verify saved checkpoint
    logger.info("\nVerifying saved checkpoint...")
    loaded_checkpoint = torch.load(save_path, map_location='cpu', weights_only=False)
    loaded_state = loaded_checkpoint['model_state_dict']
    loaded_cls_bias = loaded_state['roi_head.bbox_head.fc_cls.bias'].numpy()
    logger.info(f"Loaded background bias: {loaded_cls_bias[0]:.4f}")
    logger.info(f"Loaded foreground biases - mean: {loaded_cls_bias[1:].mean():.4f}")
    logger.info("Checkpoint created successfully!")


if __name__ == "__main__":
    main()