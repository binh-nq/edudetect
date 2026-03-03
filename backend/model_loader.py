"""
PhoBERT Detection Model Loader with Singleton Pattern
Load mô hình phát hiện AI từ HuggingFace Hub hoặc thư mục local
"""

import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import Config
from threading import Lock


class RobertaSimpleClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_labels)

    def forward(self, x, **kwargs):
        # x is the sequence output [batch, seq_len, hidden_size]
        # We take the first token (<s>) for classification
        return self.linear(x[:, 0, :])


class ModelLoader:
    _instance = None
    _lock = Lock()

    _model = None
    _tokenizer = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def _is_hub_id(self, model_path: str) -> bool:
        """Kiểm tra model_path có phải HuggingFace repo ID không (dạng 'user/repo')"""
        return '/' in model_path and not os.path.isdir(model_path)

    def load(self, model_path: str):
        """
        Load model & tokenizer từ HuggingFace Hub hoặc thư mục local (chỉ load 1 lần)

        Args:
            model_path (str): HuggingFace repo ID (vd: 'nqp426/phobert-ai-detect')
                              hoặc đường dẫn thư mục local

        Returns:
            model, tokenizer, device
        """
        if self._model is not None:
            return self._model, self._tokenizer, self._device

        if model_path is None:
            raise ValueError(
                "model_path is required. "
                "Provide a HuggingFace repo ID or local path."
            )

        is_hub = self._is_hub_id(model_path)

        # Validate local path
        if not is_hub:
            if not os.path.isdir(model_path):
                raise FileNotFoundError(f"Model directory does not exist: {model_path}")
            if not os.path.exists(os.path.join(model_path, "config.json")):
                raise FileNotFoundError("Missing config.json in model directory")

        with self._lock:
            # Double check (thread-safe)
            if self._model is not None:
                return self._model, self._tokenizer, self._device

            source = f"HuggingFace Hub: {model_path}" if is_hub else f"Local: {model_path}"
            print("=" * 60)
            print("Loading fine-tuned PhoBERT model")
            print(f"Source: {source}")

            # Device handling
            device = Config.get_device()
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            self._device = torch.device(device)
            print(f"Device: {self._device}")

            try:
                # Load tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(model_path)
                print("[OK] Tokenizer loaded")

                # Load model
                self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
                print("[OK] Model base weights loaded")

                # Load custom classifier head if exists
                if is_hub:
                    # Download classifier_head.pt từ Hub
                    self._load_classifier_head_from_hub(model_path)
                else:
                    # Load từ local
                    head_path = os.path.join(model_path, "classifier_head.pt")
                    if os.path.exists(head_path):
                        self._load_classifier_head(head_path)
                    else:
                        print("! No custom classifier head found, using default")

            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model/tokenizer from {model_path}\n"
                    f"Error: {str(e)}"
                )

            self._model.to(self._device)
            self._model.eval()

            print("[OK] Model ready for inference")
            print("=" * 60)

        return self._model, self._tokenizer, self._device

    def _load_classifier_head_from_hub(self, repo_id: str):
        """Download và load classifier_head.pt từ HuggingFace Hub"""
        try:
            from huggingface_hub import hf_hub_download
            head_path = hf_hub_download(repo_id=repo_id, filename="classifier_head.pt")
            self._load_classifier_head(head_path)
        except Exception:
            print("! No custom classifier head found on Hub, using default")

    def _load_classifier_head(self, head_path: str):
        """Load custom classifier head từ file .pt"""
        print(f"Loading custom classifier head from {head_path}")
        head_data = torch.load(head_path, map_location='cpu', weights_only=False)

        num_labels = 2
        if 'config' in head_data:
            num_labels = head_data['config'].get('num_labels', 2)

        self._model.classifier = RobertaSimpleClassifier(
            self._model.config.hidden_size,
            num_labels
        )

        state_dict = head_data.get('classifier_state_dict', head_data)
        self._model.classifier.linear.load_state_dict(state_dict)
        print("[OK] Custom classifier head loaded")

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Model chưa được load. Gọi load() trước.")
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer chưa được load. Gọi load() trước.")
        return self._tokenizer

    @property
    def device(self):
        if self._device is None:
            raise RuntimeError("Device chưa được set. Gọi load() trước.")
        return self._device
