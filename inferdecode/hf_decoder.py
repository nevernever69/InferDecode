# hf_decoder.py
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

    async def generate_full_trace(self, prompt, max_steps, temperature, top_p, top_k, decoding_strategy):
        trace = []

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated = input_ids.clone()

        for _ in range(max_steps):
            with torch.no_grad():
                outputs = self.model(input_ids=generated)
                logits = outputs.logits[:, -1, :]

            if temperature > 0:
                logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)

            if decoding_strategy == "greedy":
                next_token = torch.argmax(probs, dim=-1)
                top_probs, top_indices = torch.topk(probs, 10, dim=-1)

            elif decoding_strategy == "top_k":
                top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
                probs_top_k = top_probs / top_probs.sum(dim=-1, keepdim=True)
                sampled = torch.multinomial(probs_top_k, num_samples=1)
                next_token = torch.gather(top_indices, 1, sampled).squeeze(1)
                top_probs = probs_top_k

            elif decoding_strategy == "top_p":
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_mask = cumulative_probs <= top_p
                sorted_mask[..., 0] = True  # Ensure at least one token

                filtered_probs = sorted_probs * sorted_mask
                filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)

                sampled = torch.multinomial(filtered_probs, num_samples=1)
                next_token = torch.gather(sorted_indices, 1, sampled).squeeze(1)
                top_probs = sorted_probs[:, :10]
                top_indices = sorted_indices[:, :10]

            elif decoding_strategy == "temperature":
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                top_probs, top_indices = torch.topk(probs, 10, dim=-1)

            elif decoding_strategy == "beam_search":
                next_token = torch.topk(probs, 2, dim=-1)[1][:, 0]
                top_probs, top_indices = torch.topk(probs, 10, dim=-1)

            elif decoding_strategy == "typical":
                entropy = -(probs * probs.log()).sum(dim=-1, keepdim=True)
                surprise = (-probs.log())
                typical = (surprise - entropy).abs()
                sorted_typical, sorted_indices = torch.sort(typical, dim=-1)
                sorted_probs = torch.gather(probs, -1, sorted_indices)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_mask = cumulative_probs <= top_p
                sorted_mask[..., 0] = True

                filtered_probs = sorted_probs * sorted_mask
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

