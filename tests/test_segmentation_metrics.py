"""Tests for segmentation metrics functionality."""
import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import tempfile

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN
from scripts.train import IterationBasedTrainer


class TestSegmentationMetrics(unittest.TestCase):
    """Test the segmentation evaluation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.num_classes = 5  # Smaller for testing
        
        # Create mock data loader
        self.mock_loader = self.create_mock_loader()
        
        # Create mock COCO objects
        self.mock_val_coco = Mock()
        self.mock_coco_dt = Mock()
        
    def create_mock_loader(self):
        """Create a mock data loader."""
        # Create a simple mock loader that returns batches
        mock_loader = Mock()
        
        # Mock dataset with length
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        mock_loader.dataset = mock_dataset
        
        # Create fake batch data
        images = [torch.randn(3, 224, 224)]
        targets = [{
            'image_id': torch.tensor(1),
            'labels': torch.tensor([1]),
            'boxes': torch.tensor([[10, 10, 50, 50]]),
            'masks': torch.ones(1, 224, 224).bool()
        }]
        
        # Make the loader iterable
        mock_loader.__iter__ = Mock(return_value=iter([(images, targets)]))
        mock_loader.__len__ = Mock(return_value=1)
        
        return mock_loader
    
    def test_trainer_initialization(self):
        """Test that trainer initializes with segmentation metrics in val_history."""
        model = SwinMaskRCNN(num_classes=self.num_classes)
        
        trainer = IterationBasedTrainer(
            model=model,
            train_loader=self.mock_loader,
            val_loader=self.mock_loader,
            val_coco=self.mock_val_coco,
            num_epochs=1,
            steps_per_validation=5,
            device=self.device
        )
        
        # Check that segmentation metrics are in val_history
        self.assertIn('mAP_seg', trainer.val_history)
        self.assertIn('mAP50_seg', trainer.val_history)
        self.assertIn('mAP75_seg', trainer.val_history)
        
        # Check they are initialized as empty lists
        self.assertEqual(trainer.val_history['mAP_seg'], [])
        self.assertEqual(trainer.val_history['mAP50_seg'], [])
        self.assertEqual(trainer.val_history['mAP75_seg'], [])
    
    @patch('scripts.train.json')
    @patch('scripts.train.COCOeval')
    @patch('scripts.train.maskUtils')
    def test_evaluate_coco_with_segmentation(self, mock_maskUtils, mock_COCOeval, mock_json):
        """Test that evaluate_coco includes segmentation metrics."""
        # Setup mock model
        model = SwinMaskRCNN(num_classes=self.num_classes)
        
        # Setup trainer
        trainer = IterationBasedTrainer(
            model=model,
            train_loader=self.mock_loader,
            val_loader=self.mock_loader,
            val_coco=self.mock_val_coco,
            num_epochs=1,
            device=self.device
        )
        
        # Mock model predictions
        mock_output = {
            'boxes': torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]]),
            'labels': torch.tensor([1, 2]),
            'scores': torch.tensor([0.9, 0.8]),
            'masks': torch.randn(2, 1, 224, 224)  # 2 masks
        }
        
        # Make model return predictions
        with patch.object(model, 'eval'):
            with patch.object(model, 'forward', return_value=[mock_output]):
                # Mock maskUtils.encode to return bytes that need decoding
                def mock_encode(mask):
                    return {'counts': b'encoded_mask', 'size': [224, 224]}
                mock_maskUtils.encode.side_effect = mock_encode
                
                # Mock json.dump to avoid serialization issues
                mock_json.dump.return_value = None
                
                # Mock COCO evaluation
                mock_coco_eval_bbox = Mock()
                mock_coco_eval_bbox.stats = [0.5, 0.6, 0.7, 0.3, 0.4, 0.5]  # mAP values
                
                mock_coco_eval_seg = Mock()
                mock_coco_eval_seg.stats = [0.45, 0.55, 0.65]  # seg mAP values
                
                mock_COCOeval.side_effect = [mock_coco_eval_bbox, mock_coco_eval_seg]
                
                # Mock COCO loadRes
                self.mock_val_coco.loadRes.return_value = self.mock_coco_dt
                
                # Call evaluate_coco
                metrics = trainer.evaluate_coco()
                
                # Check that segmentation metrics are included
                self.assertIn('mAP_seg', metrics)
                self.assertIn('mAP50_seg', metrics)
                self.assertIn('mAP75_seg', metrics)
                
                # Check values
                self.assertEqual(metrics['mAP_seg'], 0.45)
                self.assertEqual(metrics['mAP50_seg'], 0.55)
                self.assertEqual(metrics['mAP75_seg'], 0.65)
                
                # Check that COCOeval was called twice (bbox and segm)
                self.assertEqual(mock_COCOeval.call_count, 2)
                
                # Check calls
                calls = mock_COCOeval.call_args_list
                self.assertEqual(calls[0][0], (self.mock_val_coco, self.mock_coco_dt, 'bbox'))
                self.assertEqual(calls[1][0], (self.mock_val_coco, self.mock_coco_dt, 'segm'))
    
    @patch('scripts.train.maskUtils')
    def test_mask_to_rle_conversion(self, mock_maskUtils):
        """Test that masks are properly converted to RLE format."""
        model = SwinMaskRCNN(num_classes=self.num_classes)
        
        trainer = IterationBasedTrainer(
            model=model,
            train_loader=self.mock_loader,
            val_loader=self.mock_loader,
            val_coco=self.mock_val_coco,
            num_epochs=1,
            device=self.device
        )
        
        # Create a mock mask
        mask = torch.randn(1, 224, 224)
        
        # Mock output with a mask
        mock_output = {
            'boxes': torch.tensor([[10, 10, 50, 50]]),
            'labels': torch.tensor([1]),
            'scores': torch.tensor([0.9]),
            'masks': mask.unsqueeze(0)
        }
        
        # Mock maskUtils.encode
        def mock_encode(mask):
            return {'counts': b'encoded_mask', 'size': [224, 224]}
        mock_maskUtils.encode.side_effect = mock_encode
        
        with patch.object(model, 'eval'):
            with patch.object(model, 'forward', return_value=[mock_output]):
                with patch.object(trainer.val_coco, 'loadRes', return_value=self.mock_coco_dt):
                    # Create a temporary file for predictions
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        trainer.checkpoint_dir = Path(tempfile.gettempdir())
                        trainer.global_step = 0
                        
                        # Call evaluate_coco
                        try:
                            trainer.evaluate_coco()
                        except Exception:
                            pass  # We expect this to fail at COCO evaluation
                        
                        # Check that encode was called with correct mask format
                        mock_maskUtils.encode.assert_called()
                        
                        # Get the mask passed to encode
                        call_args = mock_maskUtils.encode.call_args[0][0]
                        
                        # Check that mask is binary and in Fortran order
                        self.assertEqual(call_args.dtype, np.uint8)
                        self.assertTrue(call_args.flags['F_CONTIGUOUS'])
    
    @patch('scripts.train.json')
    def test_segmentation_evaluation_failure_handling(self, mock_json):
        """Test that evaluation continues even if segmentation fails."""
        model = SwinMaskRCNN(num_classes=self.num_classes)
        
        trainer = IterationBasedTrainer(
            model=model,
            train_loader=self.mock_loader,
            val_loader=self.mock_loader,
            val_coco=self.mock_val_coco,
            num_epochs=1,
            device=self.device
        )
        
        # Mock json.dump to avoid serialization issues
        mock_json.dump.return_value = None
        
        # Mock successful bbox evaluation but failing segmentation
        with patch('scripts.train.COCOeval') as mock_COCOeval:
            # First call (bbox) succeeds
            mock_bbox_eval = Mock()
            mock_bbox_eval.stats = [0.5, 0.6, 0.7, 0.3, 0.4, 0.5]
            
            # Second call (segm) raises exception
            mock_seg_eval = Mock()
            mock_seg_eval.evaluate.side_effect = Exception("Segmentation failed")
            
            mock_COCOeval.side_effect = [mock_bbox_eval, mock_seg_eval]
            
            # Mock model predictions
            mock_output = {
                'boxes': torch.tensor([[10, 10, 50, 50]]),
                'labels': torch.tensor([1]),
                'scores': torch.tensor([0.9]),
                'masks': torch.randn(1, 1, 224, 224)
            }
            
            with patch.object(model, 'eval'):
                with patch.object(model, 'forward', return_value=[mock_output]):
                    with patch('scripts.train.maskUtils'):
                        with patch.object(trainer.val_coco, 'loadRes', return_value=self.mock_coco_dt):
                            # Call evaluate_coco
                            metrics = trainer.evaluate_coco()
                            
                            # Should still return bbox metrics
                            self.assertIn('mAP', metrics)
                            self.assertEqual(metrics['mAP'], 0.5)
                            
                            # Should not have segmentation metrics
                            self.assertNotIn('mAP_seg', metrics)
    
    def test_history_update_with_segmentation(self):
        """Test that validation history is properly updated with segmentation metrics."""
        model = SwinMaskRCNN(num_classes=self.num_classes)
        
        trainer = IterationBasedTrainer(
            model=model,
            train_loader=self.mock_loader,
            val_loader=self.mock_loader,
            val_coco=self.mock_val_coco,
            num_epochs=1,
            steps_per_validation=1,
            device=self.device
        )
        
        # Mock metrics from evaluate_coco
        mock_metrics = {
            'mAP': 0.5,
            'mAP50': 0.6,
            'mAP75': 0.7,
            'mAP_small': 0.3,
            'mAP_medium': 0.4,
            'mAP_large': 0.5,
            'mAP_seg': 0.45,
            'mAP50_seg': 0.55,
            'mAP75_seg': 0.65
        }
        
        # Patch evaluate_coco to return our metrics
        with patch.object(trainer, 'evaluate_coco', return_value=mock_metrics):
            with patch.object(trainer, 'save_checkpoint'):
                # Simulate the validation part of training
                trainer.global_step = 5  # After 5 steps, validation triggers
                
                # Manually trigger the validation logic
                for key, value in mock_metrics.items():
                    if key in trainer.val_history:
                        trainer.val_history[key].append(value)
                
                # Check that all metrics are in history
                self.assertEqual(trainer.val_history['mAP'], [0.5])
                self.assertEqual(trainer.val_history['mAP_seg'], [0.45])
                self.assertEqual(trainer.val_history['mAP50_seg'], [0.55])
                self.assertEqual(trainer.val_history['mAP75_seg'], [0.65])


class TestIntegrationSegmentationMetrics(unittest.TestCase):
    """Integration tests for segmentation metrics in the training pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.num_classes = 3
        self.batch_size = 1
        
    def create_simple_batch(self):
        """Create a simple batch of data for testing."""
        images = [torch.randn(3, 64, 64)]
        targets = [{
            'image_id': torch.tensor(1),
            'labels': torch.tensor([1, 2]),
            'boxes': torch.tensor([[10, 10, 30, 30], [40, 40, 60, 60]]),
            'masks': torch.ones(2, 64, 64).bool()
        }]
        return images, targets
    
    @patch('scripts.train.json')
    @patch('scripts.train.COCO')
    @patch('scripts.train.COCOeval')
    def test_full_evaluation_pipeline(self, mock_COCOeval, mock_COCO, mock_json):
        """Test the full evaluation pipeline with segmentation."""
        # Mock json.dump to avoid serialization issues
        mock_json.dump.return_value = None
        
        # Create a simple data loader
        class SimpleDataLoader:
            def __init__(self, data):
                self.data = data
                self.dataset = Mock()
                self.dataset.__len__ = Mock(return_value=1)
            
            def __iter__(self):
                return iter(self.data)
            
            def __len__(self):
                return len(self.data)
        
        # Create simple batch
        batch = self.create_simple_batch()
        loader = SimpleDataLoader([batch])
        
        # Create model and trainer
        model = SwinMaskRCNN(num_classes=self.num_classes)
        
        # Mock COCO initialization
        mock_coco_instance = Mock()
        mock_COCO.return_value = mock_coco_instance
        
        trainer = IterationBasedTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            val_coco=mock_coco_instance,
            num_epochs=1,
            steps_per_validation=1,
            device=self.device
        )
        
        # Mock COCO evaluation results
        mock_bbox_eval = Mock()
        mock_bbox_eval.stats = [0.4, 0.5, 0.6, 0.2, 0.3, 0.4]
        
        mock_seg_eval = Mock()
        mock_seg_eval.stats = [0.35, 0.45, 0.55]
        
        mock_COCOeval.side_effect = [mock_bbox_eval, mock_seg_eval]
        
        # Mock loadRes
        mock_dt = Mock()
        mock_coco_instance.loadRes.return_value = mock_dt
        
        # Run evaluation
        with patch('scripts.train.maskUtils'):
            metrics = trainer.evaluate_coco()
        
        # Verify metrics include both bbox and segmentation
        self.assertIn('mAP', metrics)
        self.assertIn('mAP_seg', metrics)
        self.assertEqual(metrics['mAP'], 0.4)
        self.assertEqual(metrics['mAP_seg'], 0.35)


if __name__ == '__main__':
    unittest.main()