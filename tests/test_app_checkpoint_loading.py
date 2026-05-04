import pytest
import torch
import tempfile
from pathlib import Path
from deepsc_image.utils import save_checkpoint, load_checkpoint_config
from deepsc_image.app import load_model

def test_checkpoint_embedded_model_config_overrides_gui_defaults():
    # Create a dummy model and save it with specific config
    from deepsc_image.model import DeepSCImageModel
    model = DeepSCImageModel(semantic_channels=64, base_channels=16)
    
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.pth"
        save_checkpoint(path, model, {"config": {"model": {"semantic_channels": 64, "base_channels": 16}}})
        
        # Load with GUI defaults (e.g. 32, 32)
        loaded_model, status = load_model(str(path), semantic_channels=32, base_channels=32, device_name="cpu")
        
        # Should override to 64, 16
        assert loaded_model.encoder.stem.net[0].out_channels == 16 # base_channels
        assert loaded_model.encoder.project.out_channels == 64 # semantic_channels
        
        # status should mention it loaded with exact wording
        assert "C=64, base=16" in status

def test_checkpoint_loading_failure_preserves_status():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "bad.pth"
        path.write_text("not a pth file")
        
        # Load with GUI defaults
        loaded_model, status = load_model(str(path), semantic_channels=32, base_channels=32, device_name="cpu")
        
        assert status.startswith("加载 checkpoint 失败:")
        assert "未提供 checkpoint" not in status

def test_checkpoint_loading_failure_invalid_config_model():
    from deepsc_image.model import DeepSCImageModel
    model = DeepSCImageModel()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "bad_config.pth"
        save_checkpoint(path, model, {"config": {"model": "bad"}})
        
        loaded_model, status = load_model(str(path), semantic_channels=32, base_channels=32, device_name="cpu")
        
        assert status.startswith("加载 checkpoint 失败:")
        assert "Checkpoint config 'model' must be a mapping" in status

def test_load_checkpoint_config():
    from deepsc_image.model import DeepSCImageModel
    model = DeepSCImageModel()
    with tempfile.TemporaryDirectory() as d:
        # 1. Project checkpoint
        p1 = Path(d) / "proj.pth"
        save_checkpoint(p1, model, {"config": {"model": {"semantic_channels": 12, "base_channels": 8}}})
        cfg = load_checkpoint_config(p1, torch.device("cpu"))
        assert cfg == {"model": {"semantic_channels": 12, "base_channels": 8}}
        
        # 2. Raw state dict
        p2 = Path(d) / "raw.pth"
        torch.save(model.state_dict(), p2)
        cfg2 = load_checkpoint_config(p2, torch.device("cpu"))
        assert cfg2 is None

        # 3. Invalid non-mapping config
        p3 = Path(d) / "bad_config.pth"
        save_checkpoint(p3, model, {"config": "not_a_mapping"})
        with pytest.raises(ValueError, match="Checkpoint config must be a mapping"):
            load_checkpoint_config(p3, torch.device("cpu"))
