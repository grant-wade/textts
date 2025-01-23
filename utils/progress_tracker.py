from pathlib import Path
import json
import time

class FileReadingProgress:
    """Track reading progress by sentence index and save to cache file"""
    
    def __init__(self, input_path):
        self.input_path = Path(input_path)
        self.cache_path = self.input_path.with_suffix('.progress.json')
        self.current_index = 0
        self._load_progress()
        
    def _load_progress(self):
        """Load progress from cache file if it exists"""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.current_index = data.get('current_index', 0)
            except (json.JSONDecodeError, IOError):
                self.current_index = 0
                
    def save_progress(self):
        """Save current progress to cache file"""
        data = {
            'input_file': str(self.input_path),
            'current_index': self.current_index,
            'timestamp': time.time()
        }
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save progress: {e}")
            
    def update_progress(self, sentence_index):
        """Update progress and save"""
        self.current_index = sentence_index
        self.save_progress()
        
    def get_progress(self):
        """Get current progress index"""
        return self.current_index
