from typing import Any
import hashlib
import json
from pathlib import Path

class SecurityManager:
    def __init__(self):
        self.blocked_content = self._load_blocked_content()
    
    def _load_blocked_content(self):
        """Load blocked content patterns"""
        patterns_file = Path("config/blocked_patterns.json")
        if patterns_file exists():
            with open(patterns_file) as f:
                return json.load(f)
        return {"text": [], "image": []}
    
    def sanitize_input(self, data: Any) -> Any:
        """Sanitize user input"""
        if isinstance(data, str):
            # Remove potentially harmful characters
            return ''.join(char for char in data if ord(char) < 128)
        return data
    
    def check_content_safety(self, text: str = None, image_path: str = None) -> bool:
        """Check if content is safe"""
        if text:
            for pattern in self.blocked_content["text"]:
                if pattern.lower() in text.lower():
                    return False
                    
        if image_path:
            # Implement image content safety check
            pass
            
        return True
