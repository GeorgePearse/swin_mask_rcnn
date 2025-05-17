"""ONNX export callback for PyTorch Lightning."""
import torch
from pathlib import Path
from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class ONNXExportCallback(Callback):
    """Callback to export model to ONNX format during validation."""
    
    def __init__(
        self,
        export_dir: Optional[Path] = None,
        export_backbone_only: bool = True,
        save_weights: bool = True,
    ):
        """Initialize ONNX export callback.
        
        Args:
            export_dir: Directory to save ONNX files. If None, uses checkpoint directory.
            export_backbone_only: If True, exports only backbone. If False, exports full model.
            save_weights: If True, also saves raw PyTorch weights.
        """
        super().__init__()
        self.export_dir = export_dir
        self.export_backbone_only = export_backbone_only
        self.save_weights = save_weights
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Export model to ONNX at the end of validation epoch.
        
        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: PyTorch Lightning module.
        """
        # Skip if validation hasn't started yet
        if hasattr(pl_module.config, 'validation_start_step'):
            if trainer.global_step < pl_module.config.validation_start_step:
                return
        
        # Determine save directory
        if self.export_dir is None:
            # Use checkpoint directory from trainer
            checkpoint_callback = trainer.checkpoint_callback
            if checkpoint_callback and hasattr(checkpoint_callback, 'dirpath'):
                save_dir = Path(checkpoint_callback.dirpath)
            else:
                # Fallback to config checkpoint directory
                save_dir = Path(pl_module.config.checkpoint_dir)
        else:
            save_dir = self.export_dir
        
        # Ensure directory exists
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Export to ONNX
        self._export_to_onnx(pl_module, save_dir, trainer.current_epoch, trainer.global_step)
    
    def _export_to_onnx(
        self,
        pl_module: pl.LightningModule,
        save_dir: Path,
        current_epoch: int,
        global_step: int
    ) -> None:
        """Export model to ONNX format.
        
        Args:
            pl_module: PyTorch Lightning module containing the model.
            save_dir: Directory to save the ONNX file.
            current_epoch: Current training epoch.
            global_step: Current global training step.
        """
        try:
            # Create filename with epoch and step
            onnx_filename = f"model_epoch{current_epoch:03d}_step{global_step}.onnx"
            onnx_path = save_dir / onnx_filename
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 800, 800).to(pl_module.device)
            
            # Get model to export
            if self.export_backbone_only:
                model_to_export = pl_module.model.backbone
                print(f"Exporting model backbone to ONNX: {onnx_path}")
            else:
                model_to_export = pl_module.model
                print(f"Exporting full model to ONNX: {onnx_path}")
            
            # Set to eval mode
            model_to_export.eval()
            
            # Export to ONNX
            torch.onnx.export(
                model_to_export,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            print(f"Successfully exported model to ONNX: {onnx_path}")
            
            # Also save raw weights if requested
            if self.save_weights:
                weights_filename = f"weights_epoch{current_epoch:03d}_step{global_step}.pth"
                weights_path = save_dir / weights_filename
                torch.save(pl_module.model.state_dict(), weights_path)
                print(f"Saved model weights: {weights_path}")
                
        except Exception as e:
            print(f"ONNX export failed: {e}")
            # Log the error but don't crash training
            if hasattr(pl_module, 'logger') and pl_module.logger:
                pl_module.logger.log_hyperparams({"onnx_export_error": str(e)})