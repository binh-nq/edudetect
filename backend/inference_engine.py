import torch
import numpy as np
from typing import List, Dict
from config import Config
from text_processor import TextProcessor
from model_loader import ModelLoader

class InferenceEngine:
    def __init__(self, model_path: str):
        if not model_path:
            raise ValueError(
                "model_path cannot be None or empty!\n"
                "Please provide the path to your fine-tuned PhoBERT model."
            )
        
        self.loader = ModelLoader()
        self.model, self.tokenizer, self.device = self.loader.load(model_path)
        self.processor = TextProcessor()
    
    def _get_logits(self, text: str) -> torch.Tensor:
        """
        Lấy raw logits từ model
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                max_length=Config.MAX_LENGTH,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            return outputs.logits[0]  # Shape: [num_labels]

    def _get_ai_logit(self, text: str) -> float:
        """
        Lấy logit score cho class AI (index 1)
        """
        logits = self._get_logits(text)
        return logits[1].item()
    
    def _get_multi_part_scores(self, text: str) -> List[float]:
        """
        Lấy score từ 3 phần: đầu, giữa, cuối của văn bản
        """
        all_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        n = len(all_tokens)
        
        if n <= Config.MAX_LENGTH:
            logits = self._get_logits(text)
            return [torch.softmax(logits, dim=-1)[1].item()]
            
        # Xác định 3 đoạn
        segments = []
        
        # Đầu
        segments.append(all_tokens[:Config.MAX_LENGTH])
        
        # Cuối
        segments.append(all_tokens[-Config.MAX_LENGTH:])
        
        # Giữa (chỉ lấy nếu văn bản đủ dài để phần giữa không trùng hoàn toàn với đầu/cuối)
        if n > Config.MAX_LENGTH:
            mid_start = max(0, (n // 2) - (Config.MAX_LENGTH // 2))
            mid_end = min(n, mid_start + Config.MAX_LENGTH)
            segments.append(all_tokens[mid_start:mid_end])
            
        scores = []
        for seg_tokens in segments:
            # Decode ngược lại text để _get_logits xử lý (đảm bảo padding/special tokens chuẩn)
            seg_text = self.tokenizer.decode(seg_tokens, skip_special_tokens=True)
            logits = self._get_logits(seg_text)
            scores.append(torch.softmax(logits, dim=-1)[1].item())
            
        return scores

    def _batch_get_logits(self, texts: List[str]) -> List[float]:
        """
        Batch inference cho nhiều text
        
        Returns:
            List of logit scores
        """
        if not texts:
            return []
        
        logits_all = []
        
        # Process in batches
        for i in range(0, len(texts), Config.BATCH_SIZE):
            batch_texts = texts[i:i + Config.BATCH_SIZE]
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_texts,
                    max_length=Config.MAX_LENGTH,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                logits = outputs.logits[:, 1]  # AI class logits
                
                logits_all.extend(logits.cpu().numpy().tolist())
        
        return logits_all
    
    def _logit_to_prob(self, logit: float) -> float:
        """
        Chuyển raw logit sang softmax probability
        
        Sử dụng sigmoid vì chúng ta chỉ có logit của class AI
        """
        return 1 / (1 + np.exp(-logit))
    
    def _calculate_ai_score(self, sentences: List[str], sentence_results: List[Dict]) -> float:
        """
        Tính tỉ lệ số lượng từ bị đánh dấu là AI trên tổng số từ
        """
        total_words = 0
        ai_words = 0
        
        for i, sent in enumerate(sentences):
            # Đếm số từ bằng cách tách khoảng trắng
            word_count = len(sent.split())
            total_words += word_count
            if sentence_results[i]['is_ai']:
                ai_words += word_count
        
        if total_words == 0:
            return 0.0
            
        return round(ai_words / total_words, 2)

    def analyze(self, text: str) -> Dict:
        """
        Main analysis function - Sliding Window + Probability
        
        Returns:
            {
                'global_score': float,  # Phần trăm độ bao phủ AI
                'sentences': List[{
                    'text': str,
                    'score': float,
                    'is_ai': bool
                }]
            }
        """
        # Validate input
        is_valid, error = self.processor.validate_text(text)
        if not is_valid:
            raise ValueError(error)
        
        # Tách câu
        sentences = self.processor.split_sentences(text)
        
        if not sentences:
            raise ValueError("Không tìm thấy câu hợp lệ trong văn bản")
        
        # Sliding Window
        windows, sentence_map = self.processor.create_windows(
            sentences,
            Config.WINDOW_SIZE,
            Config.WINDOW_OVERLAP
        )
        
        # Batch inference cho tất cả windows
        window_texts = [' '.join(window) for window in windows]
        window_logits = self._batch_get_logits(window_texts)
        
        # Tính softmax probabilities cho từng window
        window_probs = [self._logit_to_prob(logit) for logit in window_logits]
        
        # Aggregation: probability-only
        sentence_results = []
        
        for i, sent in enumerate(sentences):
            # Lấy thông tin từ tất cả windows chứa câu này
            window_indices = sentence_map[i]
            sent_probs = [window_probs[idx] for idx in window_indices]
            
            # Max pooling
            final_prob = max(sent_probs) if sent_probs else 0.0
            
            score = float(final_prob)
            is_ai = bool(final_prob > Config.PROB_THRESHOLD)
            
            sentence_results.append({
                'text': sent,
                'score': score,
                'is_ai': is_ai
            })
        
        # Tính ai_score (phần trăm độ bao phủ AI)
        ai_score = self._calculate_ai_score(sentences, sentence_results)
        
        return {
            'global_score': ai_score,
            'sentences': sentence_results
        }
