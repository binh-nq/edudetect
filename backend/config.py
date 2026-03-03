import os

class Config:
    # Model configuration (HuggingFace Hub repo IDs)
    MODEL_PATH = 'nqp426/phobert-ai-detect'
    MAX_LENGTH = 256
    DEVICE = "cuda" 
    
    # Global Gating Thresholds
    T_HUMAN = 0.05    # → ALL_HUMAN
    T_AI = 0.95       # → ALL_AI
    
    # Sliding Window Configuration
    WINDOW_SIZE = 4          # Số câu trong mỗi window
    WINDOW_OVERLAP = 3       # Số câu overlap giữa các window
    
    # Scoring
    PROB_THRESHOLD = 0.70     # Ngưỡng probability để đánh dấu câu AI
    
    # Text Processing
    MIN_SENTENCE_LENGTH = 10  # Ký tự tối thiểu của câu hợp lệ
    
    # Inference
    BATCH_SIZE = 16         
    
    # Rewrite Model Configuration
    REWRITE_MODEL_PATH = 'nqp426/vit5-ai-rewrite'
    REWRITE_MAX_INPUT = 512   # Max input length 
    REWRITE_MAX_TARGET = 256  # Max target length
    REWRITE_NUM_BEAMS = 4     # Beam search
    
    # Sigmoid Adjustment
    SIGMOID_SCALE = 2.0      # Scaling factor cho sigmoid transformation
    
    @classmethod
    def validate_model_path(cls):
        """Kiểm tra MODEL_PATH có được set không (HF repo ID hoặc local path)"""
        if not cls.MODEL_PATH:
            raise ValueError(
                "MODEL_PATH is required!\n"
                "Set MODEL_PATH to a HuggingFace repo ID (e.g. 'nqp426/phobert-ai-detect')\n"
                "or a local directory path."
            )
    
    @classmethod
    def get_device(cls):
        import torch
        import sys
        print(f"DEBUG: sys.executable={sys.executable}")
        print(f"DEBUG: sys.version={sys.version}")
        print(f"DEBUG: Config.DEVICE={cls.DEVICE}")
        print(f"DEBUG: torch.cuda.is_available()={torch.cuda.is_available()}")
        
        if cls.DEVICE == 'cpu':
            print("DEBUG: Forcing CPU based on Config.DEVICE")
            return 'cpu'
            
        if torch.cuda.is_available():
            print("DEBUG: Selecting CUDA")
            return "cuda"
            
        print("DEBUG: Fallback to CPU because CUDA not available")
        return "cpu"