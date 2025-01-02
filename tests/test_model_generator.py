import unittest
import torch
from modules.model_generation.generator import ModelGenerator
from utils.config import Config
from pathlib import Path

class TestModelGenerator(unittest.TestCase):
    def setUp(self):
        config = Config()
        self.generator = ModelGenerator(config)
    
    def test_text_processing(self):
        test_prompt = "Generate a 3D model of a chair"
        embeddings = self.generator.process_text(test_prompt)
        
        self.assertIsInstance(embeddings, torch.Tensor)
        self.assertEqual(embeddings.dim(), 2)
    
    def test_model_generation(self):
        test_prompt = "A modern chair"
        result = self.generator.generate(test_prompt)
        
        self.assertTrue(Path(result).exists())
        self.assertEqual(Path(result).suffix, ".obj")
    
    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            self.generator.generate("")
    
    def tearDown(self):
        # Cleanup generated test files
        pass

if __name__ == '__main__':
    unittest.main()
