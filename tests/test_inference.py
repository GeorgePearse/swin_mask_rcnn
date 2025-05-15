"""Test inference functionality."""
import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image
import fiftyone as fo

from swin_maskrcnn.inference.predictor import MaskRCNNPredictor, run_inference_pipeline
from swin_maskrcnn.models.mask_rcnn import SwinMaskRCNN


class TestMaskRCNNPredictor:
    """Test MaskRCNNPredictor class."""
    
    @pytest.fixture
    def dummy_model_path(self):
        """Create a dummy model checkpoint."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            model = SwinMaskRCNN(num_classes=10)
            torch.save(model.state_dict(), f.name)
            yield f.name
            Path(f.name).unlink()
    
    @pytest.fixture
    def dummy_image_path(self):
        """Create a dummy test image."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img = Image.new('RGB', (640, 640), color='red')
            img.save(f.name)
            yield f.name
            Path(f.name).unlink()
    
    def test_predictor_initialization(self, dummy_model_path):
        """Test predictor initialization."""
        predictor = MaskRCNNPredictor(
            model_path=dummy_model_path,
            num_classes=10,
            score_threshold=0.5,
            nms_threshold=0.5
        )
        
        assert predictor.model is not None
        assert predictor.score_threshold == 0.5
        assert predictor.nms_threshold == 0.5
        assert len(predictor.class_names) == 10
    
    def test_preprocess_image(self, dummy_model_path, dummy_image_path):
        """Test image preprocessing."""
        predictor = MaskRCNNPredictor(dummy_model_path, num_classes=10)
        
        image_tensor = predictor.preprocess_image(dummy_image_path)
        
        assert isinstance(image_tensor, torch.Tensor)
        assert image_tensor.shape[0] == 3  # RGB channels
        assert image_tensor.dtype == torch.float32
    
    def test_predict_single(self, dummy_model_path, dummy_image_path):
        """Test single image prediction."""
        predictor = MaskRCNNPredictor(dummy_model_path, num_classes=10)
        
        predictions = predictor.predict_single(dummy_image_path)
        
        assert isinstance(predictions, dict)
        assert 'boxes' in predictions
        assert 'labels' in predictions
        assert 'scores' in predictions
        assert 'masks' in predictions
        
        # Check types
        assert isinstance(predictions['boxes'], np.ndarray)
        assert isinstance(predictions['labels'], np.ndarray)
        assert isinstance(predictions['scores'], np.ndarray)
    
    def test_predict_batch(self, dummy_model_path, dummy_image_path):
        """Test batch prediction."""
        predictor = MaskRCNNPredictor(dummy_model_path, num_classes=10)
        
        image_paths = [dummy_image_path] * 3
        predictions = predictor.predict_batch(image_paths)
        
        assert isinstance(predictions, list)
        assert len(predictions) == 3
        
        for pred in predictions:
            assert isinstance(pred, dict)
            assert 'boxes' in pred
            assert 'labels' in pred
            assert 'scores' in pred
    
    def test_postprocess_predictions(self, dummy_model_path):
        """Test prediction postprocessing."""
        predictor = MaskRCNNPredictor(dummy_model_path, num_classes=10)
        
        # Create dummy predictions
        predictions = {
            'boxes': torch.tensor([[10, 10, 50, 50], [100, 100, 200, 200]]),
            'labels': torch.tensor([1, 2]),
            'scores': torch.tensor([0.9, 0.8]),
            'masks': torch.rand(2, 1, 28, 28) > 0.5
        }
        
        original_size = (640, 640)
        processed = predictor.postprocess_predictions(predictions, original_size)
        
        assert isinstance(processed['boxes'], np.ndarray)
        assert processed['boxes'].shape == (2, 4)
        assert isinstance(processed['masks'], np.ndarray)
        assert processed['masks'].shape[1:] == (640, 640)  # Resized to original
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_inference(self, dummy_model_path, dummy_image_path):
        """Test GPU inference if available."""
        predictor = MaskRCNNPredictor(
            model_path=dummy_model_path,
            num_classes=10,
            device=torch.device('cuda')
        )
        
        predictions = predictor.predict_single(dummy_image_path)
        assert predictions is not None


class TestFiftyOneVisualization:
    """Test FiftyOne visualization functionality."""
    
    @pytest.fixture
    def dummy_model_path(self):
        """Create a dummy model checkpoint."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            model = SwinMaskRCNN(num_classes=10)
            torch.save(model.state_dict(), f.name)
            yield f.name
            Path(f.name).unlink()
    
    @pytest.fixture
    def dummy_image_path(self):
        """Create a dummy test image."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img = Image.new('RGB', (640, 640), color='red')
            img.save(f.name)
            yield f.name
            Path(f.name).unlink()
    
    @pytest.fixture
    def predictor(self, dummy_model_path):
        """Create predictor instance."""
        return MaskRCNNPredictor(dummy_model_path, num_classes=10)
    
    @pytest.fixture
    def dummy_predictions(self):
        """Create dummy predictions."""
        return [{
            'boxes': np.array([[10, 10, 50, 50], [100, 100, 200, 200]]),
            'labels': np.array([1, 2]),
            'scores': np.array([0.9, 0.8]),
            'masks': np.random.rand(2, 640, 640) > 0.5
        }]
    
    def test_fiftyone_dataset_creation(self, predictor, dummy_image_path, dummy_predictions):
        """Test FiftyOne dataset creation."""
        dataset = predictor.visualize_in_fiftyone(
            [dummy_image_path],
            dummy_predictions,
            dataset_name="test_dataset",
            launch_app=False
        )
        
        assert isinstance(dataset, fo.Dataset)
        assert len(dataset) == 1
        assert dataset.name == "test_dataset"
        
        # Check sample
        sample = dataset.first()
        assert sample.filepath == dummy_image_path
        assert "predictions" in sample
        assert len(sample.predictions.detections) == 2
        
        # Clean up
        dataset.delete()
    
    def test_detection_format(self, predictor, dummy_image_path, dummy_predictions):
        """Test detection format in FiftyOne."""
        dataset = predictor.visualize_in_fiftyone(
            [dummy_image_path],
            dummy_predictions,
            launch_app=False
        )
        
        sample = dataset.first()
        detection = sample.predictions.detections[0]
        
        assert hasattr(detection, 'label')
        assert hasattr(detection, 'bounding_box')
        assert hasattr(detection, 'confidence')
        assert hasattr(detection, 'mask')
        
        # Check bounding box format (relative coordinates)
        assert all(0 <= x <= 1 for x in detection.bounding_box)
        
        # Clean up
        dataset.delete()


class TestInferencePipeline:
    """Test complete inference pipeline."""
    
    @pytest.fixture
    def dummy_model_path(self):
        """Create a dummy model checkpoint."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            model = SwinMaskRCNN(num_classes=10)
            torch.save(model.state_dict(), f.name)
            yield f.name
            Path(f.name).unlink()
    
    @pytest.fixture
    def dummy_image_path(self):
        """Create a dummy test image."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img = Image.new('RGB', (640, 640), color='red')
            img.save(f.name)
            yield f.name
            Path(f.name).unlink()
    
    def test_run_inference_pipeline(self, dummy_model_path, dummy_image_path):
        """Test complete inference pipeline."""
        results = run_inference_pipeline(
            model_path=dummy_model_path,
            image_paths=[dummy_image_path],
            num_classes=10,
            visualize=False
        )
        
        assert 'predictions' in results
        assert 'image_paths' in results
        assert len(results['predictions']) == 1
        assert results['image_paths'] == [dummy_image_path]
    
    def test_pipeline_with_visualization(self, dummy_model_path, dummy_image_path):
        """Test pipeline with FiftyOne visualization."""
        results = run_inference_pipeline(
            model_path=dummy_model_path,
            image_paths=[dummy_image_path],
            num_classes=10,
            visualize=True,
            dataset_name="test_pipeline"
        )
        
        assert 'fiftyone_dataset' in results
        dataset = results['fiftyone_dataset']
        
        if isinstance(dataset, tuple):  # Returns (dataset, session) when launch_app=True
            dataset = dataset[0]
        
        assert isinstance(dataset, fo.Dataset)
        assert len(dataset) == 1
        
        # Clean up
        dataset.delete()
    
    def test_pipeline_single_image_string(self, dummy_model_path, dummy_image_path):
        """Test pipeline with single image path as string."""
        results = run_inference_pipeline(
            model_path=dummy_model_path,
            image_paths=dummy_image_path,  # Single string, not list
            num_classes=10,
            visualize=False
        )
        
        assert isinstance(results['image_paths'], list)
        assert len(results['image_paths']) == 1
        assert results['image_paths'][0] == dummy_image_path


def test_imports():
    """Test that all required packages can be imported."""
    import fiftyone
    import torch
    import torchvision
    import numpy
    import PIL
    
    # Check FiftyOne is properly installed
    assert hasattr(fiftyone, 'Dataset')
    assert hasattr(fiftyone, 'launch_app')