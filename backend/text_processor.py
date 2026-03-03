from underthesea import sent_tokenize
from typing import List
from config import Config

class TextProcessor:
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        # Tách câu
        sentences = sent_tokenize(text)
        
        # Lọc câu quá ngắn hoặc rỗng
        valid_sentences = [
            sent.strip() 
            for sent in sentences 
            if len(sent.strip()) >= Config.MIN_SENTENCE_LENGTH
        ]
        
        return valid_sentences
    
    @staticmethod
    def create_windows(sentences: List[str], window_size: int, overlap: int) -> tuple:
        if len(sentences) <= window_size:
            # Văn bản ngắn: coi toàn bộ là 1 window
            return [sentences], {i: [0] for i in range(len(sentences))}
        
        windows = []
        sentence_map = {i: [] for i in range(len(sentences))}
        
        step = window_size - overlap
        start = 0
        
        while start < len(sentences):
            end = min(start + window_size, len(sentences))
            window = sentences[start:end]
            window_idx = len(windows)
            windows.append(window)
            
            # Cập nhật sentence_map
            for i in range(start, end):
                sentence_map[i].append(window_idx)
            
            if end == len(sentences):
                break
            start += step
        
        return windows, sentence_map
    
    @staticmethod
    def validate_text(text: str) -> tuple:
        if not text or not text.strip():
            return False, "Văn bản không được để trống"
        
        if len(text.strip()) < Config.MIN_SENTENCE_LENGTH:
            return False, f"Văn bản quá ngắn (tối thiểu {Config.MIN_SENTENCE_LENGTH} ký tự)"
        
        return True, None