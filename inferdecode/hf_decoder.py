# hf_decoder.py (updated implementation)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from inferdecode.base_decoder import BaseDecoder
from huggingface_hub import login


class HFDecoder(BaseDecoder):
    def __init__(self, model_name: str, device="cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if device=="cuda" else torch.float32
        )
        self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @property
    def model_name(self) -> str:
        return self.model.name_or_path

    async def generate_full_trace(self, prompt, max_steps, temperature, top_p, top_k, decoding_strategy):
        trace = []
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated = input_ids.clone()

        # Validate parameters
        top_p = max(0.0, min(1.0, top_p))
        top_k = max(1, top_k)
        temperature = max(0.0, temperature)

        for _ in range(max_steps):
            with torch.no_grad():
                outputs = self.model(input_ids=generated)
                logits = outputs.logits[:, -1, :]

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)

            if decoding_strategy == "greedy":
                next_token = torch.argmax(probs, dim=-1)
                top_probs, top_indices = torch.topk(probs, 10, dim=-1)

            elif decoding_strategy == "top_k":
                # Ensure top_k doesn't exceed vocabulary size
                vocab_size = probs.size(-1)
                actual_top_k = min(top_k, vocab_size)
                
                top_probs, top_indices = torch.topk(probs, actual_top_k, dim=-1)
                probs_top_k = top_probs / top_probs.sum(dim=-1, keepdim=True)
                sampled = torch.multinomial(probs_top_k, num_samples=1)
                next_token = torch.gather(top_indices, 1, sampled).squeeze(1)
                # Get top 10 for display
                top_probs, top_indices = torch.topk(probs, 10, dim=-1)

            elif decoding_strategy == "top_p":
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Find the first index where cumulative probability exceeds top_p
                sorted_mask = cumulative_probs <= top_p
                # Ensure at least one token is selected
                sorted_mask[..., 0] = True  

                # Remove tokens that don't meet the criteria
                filtered_probs = torch.where(sorted_mask, sorted_probs, torch.zeros_like(sorted_probs))
                filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)

                sampled = torch.multinomial(filtered_probs, num_samples=1)
                next_token = torch.gather(sorted_indices, 1, sampled).squeeze(1)
                top_probs = sorted_probs[:, :10]
                top_indices = sorted_indices[:, :10]

            elif decoding_strategy == "temperature":
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                top_probs, top_indices = torch.topk(probs, 10, dim=-1)

            elif decoding_strategy == "beam_search":
                # Note: This is a simplified version - true beam search requires maintaining multiple sequences
                # For research purposes, we'll show the top 2 candidates at each step
                top_probs, top_indices = torch.topk(probs, 2, dim=-1)
                next_token = top_indices[:, 0]  # Take the most probable token
                # Get top 10 for display
                top_probs, top_indices = torch.topk(probs, 10, dim=-1)

            elif decoding_strategy == "typical":
                # Calculate entropy
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1, keepdim=True)
                # Calculate surprise (negative log probability)
                surprise = -torch.log(probs + 1e-10)
                # Calculate typicality
                typical = (surprise - entropy).abs()
                
                # Sort by typicality
                sorted_typical, sorted_indices = torch.sort(typical, dim=-1)
                sorted_probs = torch.gather(probs, -1, sorted_indices)
                
                # Apply top-p filtering
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_mask = cumulative_probs <= top_p
                sorted_mask[..., 0] = True  # Ensure at least one token

                filtered_probs = torch.where(sorted_mask, sorted_probs, torch.zeros_like(sorted_probs))
                filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)

                sampled = torch.multinomial(filtered_probs, num_samples=1)
                next_token = torch.gather(sorted_indices, 1, sampled).squeeze(1)
                top_probs, top_indices = torch.topk(probs, 10, dim=-1)

            else:
                raise ValueError(f"Unknown decoding strategy {decoding_strategy}")

            token_str = self.tokenizer.decode(next_token[0])
            full_text = self.tokenizer.decode(torch.cat([generated[0], next_token]).tolist())

            trace.append({
                "step": len(trace) + 1,
                "top_tokens": [
                    {"token": self.tokenizer.decode([idx.item()]), "prob": prob.item()}
                    for idx, prob in zip(top_indices[0], top_probs[0])
                ],
                "chosen_token": token_str,
                "current_text": full_text
            })

            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return trace
