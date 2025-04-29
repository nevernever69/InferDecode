from inferdecode.hf_decoder import HFDecoder
from inferdecode.decode_tui import DecodeTUI

def main():
    import argparse

    parser = argparse.ArgumentParser(description="InferDecode: Interactive Decoding Visualizer")
    parser.add_argument("--model", type=str, default="openai-community/gpt2", help="Model name from Hugging Face")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Device to use")
    args = parser.parse_args()

    decoder = HFDecoder(model_name=args.model, device=args.device)
    app = DecodeTUI(decoder)
    app.run()

if __name__ == "__main__":
    main()

