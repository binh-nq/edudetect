import os

class Config:
    # Model configuration 
    MODEL_PATH = 'nqp426/phobert-ai-detect'
    MAX_LENGTH = 256
    DEVICE = "cuda" 
    # Sliding Window Configuration
    WINDOW_SIZE = 4         
    WINDOW_OVERLAP = 3      
    
    # Scoring
    PROB_THRESHOLD = 0.70     
    
    # Text Processing
    MIN_SENTENCE_LENGTH = 10  #
    
    # Inference
    BATCH_SIZE = 16         
    
    # Rewrite Model Configuration
    REWRITE_MODEL_PATH = 'nqp426/vit5-ai-rewrite'
    REWRITE_MAX_INPUT = 512   
    REWRITE_MAX_TARGET = 256 
    REWRITE_NUM_BEAMS = 4     
    
    # Sigmoid Adjustment
    SIGMOID_SCALE = 2.0      
    
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