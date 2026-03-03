import torch
from typing import Optional
from config import Config
from rewrite_loader import RewriteModelLoader


class RewriteEngine:
    def __init__(self, model_path: str = None):
        self.loader = RewriteModelLoader()
        self.model, self.tokenizer, self.device = self.loader.load(model_path)
    
    def _format_input(
        self,
        mode: str,
        target: str,
        prev_context: Optional[str] = None,
        next_context: Optional[str] = None
    ) -> str:
        """
        Format input theo cấu trúc đã train:
        [FIX] prev_context <target>sentence</target> next_context
        hoặc
        [REWRITE] text
        """
        mode_token = "[FIX]" if mode == "fix" else "[REWRITE]"
        
        if mode == "rewrite":
            return f"{mode_token} {target}"
        
        parts = [mode_token]
        
        if prev_context and prev_context.strip():
            parts.append(prev_context.strip())
        
        parts.append(f"<target>{target.strip()}</target>")
        
        if next_context and next_context.strip():
            parts.append(next_context.strip())
        
        return " ".join(parts)
    
    def rewrite(
        self,
        target: str,
        mode: str = "fix",
        prev_context: Optional[str] = None,
        next_context: Optional[str] = None
    ) -> str:
        """
        Viết lại văn bản
        
        Args:
            target: Câu/đoạn cần viết lại
            mode: "fix" (1 câu với context) hoặc "rewrite" (cả đoạn)
            prev_context: Câu trước (optional, chỉ dùng cho mode fix)
            next_context: Câu sau (optional, chỉ dùng cho mode fix)
            
        Returns:
            Văn bản đã được viết lại
        """
        if not target or not target.strip():
            raise ValueError("Target text không được để trống")
        
        # Validate mode
        mode = mode.lower()
        if mode not in ["fix", "rewrite"]:
            raise ValueError("Mode phải là 'fix' hoặc 'rewrite'")
        
        # Format input
        input_text = self._format_input(mode, target, prev_context, next_context)
        
        # Tokenize
        with torch.no_grad():
            inputs = self.tokenizer(
                input_text,
                max_length=Config.REWRITE_MAX_INPUT,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=Config.REWRITE_MAX_TARGET,
                num_beams=Config.REWRITE_NUM_BEAMS,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            
            # Decode
            rewritten = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return rewritten.strip()
    
    def rewrite_sentence(
        self,
        sentence: str,
        prev_context: Optional[str] = None,
        next_context: Optional[str] = None
    ) -> str:

        return self.rewrite(
            target=sentence,
            mode="fix",
            prev_context=prev_context,
            next_context=next_context
        )
    
    def rewrite_paragraph(self, text: str) -> str:

        return self.rewrite(target=text, mode="rewrite")
