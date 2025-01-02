import unittest
import torch
from modules.nlp.text_processor import TextProcessor
from modules.image_processing.image_processor import ImageProcessor

class TestProcessors(unittest.TestCase):
    def setUp(self):
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
    
    def test_text_processor(self):
        text = "Test metni"
        result = self.text_processor.process(text)
        self.assertIsInstance(result, torch.Tensor)
    
    def test_image_processor(self):
        # Test görselini oluştur
        dummy_image = torch.randn(3, 224, 224)
        result = self.image_processor.process(dummy_image)
        self.assertIsInstance(result, torch.Tensor)

if __name__ == '__main__':
    unittest.main() 