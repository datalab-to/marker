import unittest
import sys
import os

# Add the project root to the path
sys.path.insert(0, '.')

from marker.models import create_model_dict
from marker.config.parser import ConfigParser


class TestModelIntegration(unittest.TestCase):
    def test_model_creation(self):
        """Test model creation"""
        # Test model creation
        models = create_model_dict(device="cpu")
        
        # Verify models are created
        self.assertIn("layout_model", models)
        self.assertIn("recognition_model", models)
        self.assertIn("foundation_model", models)
        self.assertIn("table_rec_model", models)
        self.assertIn("detection_model", models)
        self.assertIn("ocr_error_model", models)

    def test_cli_options_integration(self):
        """Test CLI options integration for model_workers (should be ignored)"""
        # Test model_workers CLI option (should be ignored)
        cli_options = {
            "model_workers": "layout_model:2,recognition_model:3"
        }
        
        config_parser = ConfigParser(cli_options)
        config_dict = config_parser.generate_config_dict()
        
        # Verify that model_workers doesn't create an entry
        self.assertNotIn("model_workers", config_dict)


if __name__ == '__main__':
    unittest.main()