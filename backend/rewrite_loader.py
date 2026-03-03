import os
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast, T5ForConditionalGeneration
from config import Config
from threading import Lock


class RewriteModelLoader:
    _instance = None
    _lock = Lock()

    _model = None
    _tokenizer = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RewriteModelLoader, cls).__new__(cls)
        return cls._instance

    def _is_hub_id(self, model_path: str) -> bool:
        return '/' in model_path and not os.path.isdir(model_path)

    def load(self, model_path: str = None):
        if self._model is not None:
            return self._model, self._tokenizer, self._device

        if model_path is None:
            model_path = Config.REWRITE_MODEL_PATH

        is_hub = self._is_hub_id(model_path)

        if not is_hub:
            if not os.path.isdir(model_path):
                raise FileNotFoundError(f"Rewrite model directory does not exist: {model_path}")
            if not os.path.exists(os.path.join(model_path, "config.json")):
                raise FileNotFoundError("Missing config.json in rewrite model directory")

        with self._lock:
            if self._model is not None:
                return self._model, self._tokenizer, self._device

            source = f"HuggingFace Hub: {model_path}" if is_hub else f"Local: {model_path}"
            print("=" * 60)
            print("Loading fine-tuned ViT5 rewrite model")
            print(f"Source: {source}")

            device = Config.get_device()
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            self._device = torch.device(device)
            print(f"Device: {self._device}")

            try:
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(model_path)
                except (KeyError, Exception):
                    if is_hub:
                        from huggingface_hub import hf_hub_download
                        tokenizer_file = hf_hub_download(repo_id=model_path, filename="tokenizer.json")
                        self._tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
                    else:
                        tokenizer_file = os.path.join(model_path, "tokenizer.json")
                        if os.path.exists(tokenizer_file):
                            self._tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
                        else:
                            raise
                print("[OK] Tokenizer loaded")

                self._model = T5ForConditionalGeneration.from_pretrained(model_path)
                print("[OK] Model weights loaded")

            except Exception as e:
                raise RuntimeError(
                    f"Failed to load rewrite model/tokenizer from {model_path}\n"
                    f"Error: {str(e)}"
                )

            self._model.to(self._device)
            self._model.eval()

            print("[OK] Rewrite model ready for inference")
            print("=" * 60)

        return self._model, self._tokenizer, self._device

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Rewrite model chưa được load. Gọi load() trước.")
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
