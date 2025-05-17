"""Test ONNX export functionality."""
import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from scripts.train import MaskRCNNLightningModule
from scripts.config import TrainingConfig


class MockLightningModule:
    """Mock version of MaskRCNNLightningModule for testing."""
    
    def __init__(self, config):
        self.config = config
        self.current_epoch = 1
        self.global_step = 100
        self.device = torch.device('cpu')
        self.model = Mock()
        self.model.state_dict = Mock(return_value={})
        self.model.backbone = Mock()
        self.model.eval = Mock()
    
    def export_to_onnx(self, save_dir: Path):
        """Mock implementation of export_to_onnx."""
        # Mimic the actual implementation
        from scripts.train import MaskRCNNLightningModule
        MaskRCNNLightningModule.export_to_onnx(self, save_dir)


class TestONNXExport:
    """Test ONNX export functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return TrainingConfig(
            num_classes=69,
            checkpoint_dir=Path(tempfile.mkdtemp())
        )
    
    @pytest.fixture
    def mock_module(self, config):
        """Create mock Lightning module for testing."""
        return MockLightningModule(config)
    
    def test_export_to_onnx(self, mock_module, config):
        """Test ONNX export method."""
        save_dir = config.checkpoint_dir / "test_run"
        save_dir.mkdir(exist_ok=True)
        
        # Mock torch.onnx.export
        with patch('torch.onnx.export') as mock_export, \
             patch('torch.save') as mock_save:
            
            # Test export
            mock_module.export_to_onnx(save_dir)
            
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
    
    def test_validation_epoch_end_exports(self):
        """Test that validation epoch end triggers ONNX export."""
        config = TrainingConfig(
            num_classes=69,
            checkpoint_dir=Path(tempfile.mkdtemp())
        )
        
        # Create a real module instance for this test
        val_coco = Mock()
        module = MaskRCNNLightningModule(config=config, val_coco=val_coco)
        
        # Mock necessary properties and methods
        module.trainer = Mock()
        checkpoint_callback = Mock()
        checkpoint_callback.dirpath = str(config.checkpoint_dir / "test_run")
        module.trainer.checkpoint_callback = checkpoint_callback
        
        # Create the directory
        Path(checkpoint_callback.dirpath).mkdir(exist_ok=True)
        
        # Mock validation outputs
        module.validation_outputs = []
        
        # Mock global_step property
        with patch.object(type(module), 'global_step', new_callable=lambda: property(lambda self: 1000)):
            # Mock the export method
            with patch.object(module, 'export_to_onnx') as mock_export:
                module.on_validation_epoch_end()
                
                # Check that export was called
                mock_export.assert_called_once()
                save_dir = mock_export.call_args[0][0]
                assert isinstance(save_dir, Path)
                assert str(save_dir) == checkpoint_callback.dirpath
    
    def test_export_handles_errors(self, mock_module):
        """Test that export handles errors gracefully."""
        save_dir = Path("/nonexistent/directory")
        
        # Should not raise exception
        mock_module.export_to_onnx(save_dir)
        
        # Verify no files were created
        assert not save_dir.exists()
    
    def test_export_filename_format(self, config):
        """Test that export filenames follow the correct format."""
        save_dir = config.checkpoint_dir / "test_format"
        save_dir.mkdir(exist_ok=True)
        
        # Create module with specific epoch and step values
        module = MockLightningModule(config)
        module.current_epoch = 10
        module.global_step = 5000
        
        # Mock torch functions
        with patch('torch.onnx.export') as mock_export, \
             patch('torch.save') as mock_save:
            
            # Export
            module.export_to_onnx(save_dir)
            
            # Verify filenames
            export_args = mock_export.call_args
            save_args = mock_save.call_args
            
            onnx_path = Path(export_args[0][2])
            assert onnx_path.name == "model_epoch010_step5000.onnx"
            
            weights_path = Path(save_args[0][1])
            assert weights_path.name == "weights_epoch010_step5000.pth"
    
    def test_integration_with_trainer(self):
        """Test integration with PyTorch Lightning trainer."""
        config = TrainingConfig(
            num_classes=69,
            checkpoint_dir=Path(tempfile.mkdtemp()),
            validation_start_step=0  # Enable validation immediately
        )
        
        # Create real module
        val_coco = Mock()
        module = MaskRCNNLightningModule(config=config, val_coco=val_coco)
        
        # Mock trainer
        module.trainer = Mock()
        checkpoint_callback = Mock()
        checkpoint_callback.dirpath = str(config.checkpoint_dir / "test_run")
        module.trainer.checkpoint_callback = checkpoint_callback
        
        # Create directory
        Path(checkpoint_callback.dirpath).mkdir(exist_ok=True)
        
        # Mock validation outputs
        module.validation_outputs = []
        
        # Mock global_step property
        with patch.object(type(module), 'global_step', new_callable=lambda: property(lambda self: 500)):
            # Mock the actual ONNX export and weight saving
            with patch('torch.onnx.export'), \
                 patch('torch.save'):
                
                # Should execute without errors
                module.on_validation_epoch_end()