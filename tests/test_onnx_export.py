"""Test ONNX export callback functionality."""
import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytorch_lightning as pl

from swin_maskrcnn.callbacks.onnx_export import ONNXExportCallback
from scripts.train import MaskRCNNLightningModule
from scripts.config import TrainingConfig


class TestONNXExportCallback:
    """Test ONNX export callback functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return TrainingConfig(
            num_classes=69,
            checkpoint_dir=Path(tempfile.mkdtemp())
        )
    
    @pytest.fixture
    def callback(self, config):
        """Create ONNX export callback."""
        return ONNXExportCallback(
            export_dir=config.checkpoint_dir / "test_run",
            export_backbone_only=True,
            save_weights=True
        )
    
    @pytest.fixture
    def mock_trainer(self):
        """Create mock trainer."""
        trainer = Mock(spec=pl.Trainer)
        trainer.current_epoch = 1
        trainer.global_step = 100
        checkpoint_callback = Mock()
        checkpoint_callback.dirpath = str(tempfile.mkdtemp())
        trainer.checkpoint_callback = checkpoint_callback
        return trainer
    
    @pytest.fixture
    def mock_pl_module(self, config):
        """Create mock Lightning module."""
        module = Mock(spec=MaskRCNNLightningModule)
        module.config = config
        module.device = torch.device('cpu')
        module.model = Mock()
        module.model.backbone = Mock()
        module.model.state_dict = Mock(return_value={})
        return module
    
    def test_callback_initialization(self):
        """Test callback initialization with various parameters."""
        # Test with default parameters
        callback = ONNXExportCallback()
        assert callback.export_dir is None
        assert callback.export_backbone_only is True
        assert callback.save_weights is True
        
        # Test with custom parameters
        export_dir = Path("/tmp/test")
        callback = ONNXExportCallback(
            export_dir=export_dir,
            export_backbone_only=False,
            save_weights=False
        )
        assert callback.export_dir == export_dir
        assert callback.export_backbone_only is False
        assert callback.save_weights is False
    
    def test_export_to_onnx(self, callback, mock_trainer, mock_pl_module):
        """Test ONNX export functionality."""
        callback.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock torch.onnx.export
        with patch('torch.onnx.export') as mock_export, \
             patch('torch.save') as mock_save:
            
            # Test export
            callback._export_to_onnx(
                mock_pl_module,
                callback.export_dir,
                current_epoch=1,
                global_step=100
            )
            
            # Check that export was called
            mock_export.assert_called_once()
            
            # Check that weights were saved
            mock_save.assert_called_once()
            
            # Verify filenames in the calls
            export_args = mock_export.call_args
            save_args = mock_save.call_args
            
            # ONNX filename should match pattern
            onnx_path = Path(export_args[0][2])
            assert onnx_path.name == "model_epoch001_step100.onnx"
            
            # Weights filename should match pattern  
            weights_path = Path(save_args[0][1])
            assert weights_path.name == "weights_epoch001_step100.pth"
    
    def test_on_validation_epoch_end(self, callback, mock_trainer, mock_pl_module):
        """Test that validation epoch end triggers ONNX export."""
        # Create directory
        callback.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Set validation start step
        mock_pl_module.config.validation_start_step = 0
        
        # Mock the export method
        with patch.object(callback, '_export_to_onnx') as mock_export:
            callback.on_validation_epoch_end(mock_trainer, mock_pl_module)
            
            # Check that export was called
            mock_export.assert_called_once_with(
                mock_pl_module,
                callback.export_dir,
                mock_trainer.current_epoch,
                mock_trainer.global_step
            )
    
    def test_skip_export_before_validation_start(self, callback, mock_trainer, mock_pl_module):
        """Test that export is skipped before validation starts."""
        # Set validation start step higher than current step
        mock_pl_module.config.validation_start_step = 200
        mock_trainer.global_step = 50
        
        # Mock the export method
        with patch.object(callback, '_export_to_onnx') as mock_export:
            callback.on_validation_epoch_end(mock_trainer, mock_pl_module)
            
            # Check that export was NOT called
            mock_export.assert_not_called()
    
    def test_export_handles_errors(self, callback, mock_trainer, mock_pl_module):
        """Test that export handles errors gracefully."""
        # Make export fail
        with patch('torch.onnx.export', side_effect=Exception("Export failed")):
            # Should not raise exception
            callback._export_to_onnx(
                mock_pl_module,
                Path("/nonexistent/directory"),
                current_epoch=1,
                global_step=100
            )
    
    def test_filename_format(self, callback):
        """Test that export filenames follow the correct format."""
        save_dir = callback.export_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock module
        module = Mock()
        module.device = torch.device('cpu')
        module.model = Mock()
        module.model.backbone = Mock()
        module.model.state_dict = Mock(return_value={})
        
        # Mock torch functions
        with patch('torch.onnx.export') as mock_export, \
             patch('torch.save') as mock_save:
            
            # Export with specific epoch and step
            callback._export_to_onnx(module, save_dir, current_epoch=10, global_step=5000)
            
            # Verify filenames
            export_args = mock_export.call_args
            save_args = mock_save.call_args
            
            onnx_path = Path(export_args[0][2])
            assert onnx_path.name == "model_epoch010_step5000.onnx"
            
            weights_path = Path(save_args[0][1])
            assert weights_path.name == "weights_epoch010_step5000.pth"
    
    def test_integration_with_lightning(self, config):
        """Test integration with PyTorch Lightning training."""
        # Create callback
        callback = ONNXExportCallback(export_dir=config.checkpoint_dir / "test_run")
        
        # Create mock module and trainer
        val_coco = Mock()
        module = MaskRCNNLightningModule(config=config, val_coco=val_coco)
        
        trainer = Mock(spec=pl.Trainer)
        trainer.current_epoch = 1
        trainer.global_step = 500
        checkpoint_callback = Mock()
        checkpoint_callback.dirpath = str(config.checkpoint_dir / "test_run")
        trainer.checkpoint_callback = checkpoint_callback
        
        # Mock the actual ONNX export and weight saving
        with patch('torch.onnx.export'), \
             patch('torch.save'):
            
            # Should execute without errors
            callback.on_validation_epoch_end(trainer, module)