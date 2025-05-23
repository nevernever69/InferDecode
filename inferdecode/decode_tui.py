from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Input, Button, Static, Select
from textual.worker import Worker, get_current_worker, WorkerCancelled
from textual.message import Message
import asyncio
import time
from rich.panel import Panel
from rich.text import Text

# Custom message to update the UI with each token
class TokenUpdate(Message):
    def __init__(self, step_info, cur_pos, total_steps, total_time, avg_time_per_token):
        super().__init__()
        self.step_info = step_info
        self.cur_pos = cur_pos
        self.total_steps = total_steps
        self.total_time = total_time
        self.avg_time_per_token = avg_time_per_token

class DecodeTUI(App):
    CSS = """ 
    Screen { layout: vertical; align: center top; padding: 1; }
    #settings_bar, #buttons_bar { layout: horizontal; height: auto; align: center top; padding: 1; }
    #visuals { layout: horizontal; align: center top; height: 1fr; width: 100%; margin-top: 2; }
    .small_input, .small_button { width: 12%; height: 3; margin: 0 1; }
    .prompt_input { width: 100%; height: 3; margin: 1 1; }
    #tokens, #generated, #metrics { border: heavy $accent; padding: 1; width: 1fr; height: 1fr; }
    """

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.prompt = ""
        self.decoding_strategy = "greedy"
        self.top_p = 0.9
        self.top_k = 40
        self.temperature = 1.0
        self.max_steps = 64
        self.delay = 0.3

        self.trace = []
        self.cur_pos = 0
        self.finished = False
        self.stop_requested = False
        self.total_time = 0
        self.avg_time_per_token = 0
        self.trace_prompt = None
        self.paused = False
        self.worker = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="settings_bar"):
            yield Select(
                options=[
                    ("Greedy", "greedy"),
                    ("Top-k Sampling", "top_k"),
                    ("Top-p Sampling", "top_p"),
                    ("Temperature Sampling", "temperature"),
                    ("Beam Search", "beam_search"),
                    ("Typical Decoding (Top-A)", "typical"),
                ],
                value="greedy",
                id="strategy",
                classes="small_input"
            )
            yield Input(placeholder="Top-p (0.9)", id="top_p", classes="small_input")
            yield Input(placeholder="Top-k (40)", id="top_k", classes="small_input")
            yield Input(placeholder="Temp (1.0)", id="temperature", classes="small_input")
            yield Input(placeholder="Steps (64)", id="max_steps", classes="small_input")
            yield Input(placeholder="Delay (0.3s)", id="delay", classes="small_input")

        yield Input(placeholder="Enter your prompt here...", id="prompt", classes="prompt_input")

        with Horizontal(id="buttons_bar"):
            yield Button("Start ➡️", id="start", classes="small_button")
            yield Button("Step ⬇️", id="step", classes="small_button")
            yield Button("Stop ⏹", id="stop", classes="small_button")

        with Horizontal(id="visuals"):
            yield Static("Top Candidates", id="tokens")
            yield Static("Generated Text", id="generated")
            yield Static("Metrics 📈", id="metrics")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self.stop_requested = False
            await self.start_decoding(full_run=True)
        elif event.button.id == "step":
            self.stop_requested = False
            await self.start_decoding(full_run=False)
        elif event.button.id == "stop":
            self.stop_requested = True
            if self.worker and not self.worker.is_cancelled:
                self.worker.cancel()
                try:
                    await self.worker.wait()  # Wait for the worker to fully stop
                except WorkerCancelled:
                    pass  # Ignore the cancellation exception
                self.finished = True  # Mark as finished to prevent further generation

    async def start_decoding(self, full_run=False):
        current_prompt = self.query_one("#prompt", Input).value
        if (self.trace_prompt != current_prompt) or not self.trace or self.finished:
            # Generate new trace
            self.trace_prompt = current_prompt
            self.prompt = current_prompt
            self.decoding_strategy = self.query_one("#strategy", Select).value
            self.top_p = float(self.query_one("#top_p", Input).value or 0.9)
            self.top_k = int(self.query_one("#top_k", Input).value or 40)
            self.temperature = float(self.query_one("#temperature", Input).value or 1.0)
            self.max_steps = int(self.query_one("#max_steps", Input).value or 64)
            self.delay = float(self.query_one("#delay", Input).value or 0.3)

            self.cur_pos = 0
            self.finished = False
            self.paused = False
            self.stop_requested = False

            start_time = time.time()
            self.trace = await self.model.generate_full_trace(
                prompt=self.prompt,
                max_steps=self.max_steps,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                decoding_strategy=self.decoding_strategy
            )
            self.total_time = time.time() - start_time
            self.avg_time_per_token = self.total_time / max(1, len(self.trace))
        else:
            # Continue existing trace
            self.stop_requested = False
            self.paused = False

        # Start the worker for full generation or process one step
        if full_run and not self.stop_requested:
            self.worker = self.run_worker(self.run_full_generation(), exclusive=True)
        else:
            await self.decode_one_token()

    async def run_full_generation(self):
        """Worker method to run the generation loop."""
        worker = get_current_worker()
        while not self.finished and not self.stop_requested and self.cur_pos < len(self.trace):
            if worker.is_cancelled:
                break  # Exit immediately if canceled
            step_info = self.trace[self.cur_pos]
            self.post_message(TokenUpdate(step_info, self.cur_pos, len(self.trace), 
                                        self.total_time, self.avg_time_per_token))
            self.cur_pos += 1
            if self.cur_pos >= len(self.trace):
                self.finished = True
            await asyncio.sleep(self.delay)

    async def decode_one_token(self):
        """Process a single token for the Step button."""
        if self.finished or self.cur_pos >= len(self.trace) or self.stop_requested:
            self.finished = True
            return
        step_info = self.trace[self.cur_pos]
        self.post_message(TokenUpdate(step_info, self.cur_pos, len(self.trace), 
                                    self.total_time, self.avg_time_per_token))
        self.cur_pos += 1
        if self.cur_pos >= len(self.trace):
            self.finished = True

    async def on_token_update(self, message: TokenUpdate):
        """Handle UI updates from the worker."""
        step_info = message.step_info
        cur_pos = message.cur_pos
        total_steps = message.total_steps
        total_time = message.total_time
        avg_time_per_token = message.avg_time_per_token

        tokens_panel = self.query_one("#tokens", Static)
        gen_panel = self.query_one("#generated", Static)
        metrics_panel = self.query_one("#metrics", Static)

        token_text = Text("", justify="left")
        max_prob = step_info["top_tokens"][0]['prob'] if step_info["top_tokens"] else 1.0
        for tok_info in step_info["top_tokens"][:10]:
            token = tok_info['token']
            prob = tok_info['prob']
            width = int(prob / max_prob * 30)
            bar = "█" * width
            token_text.append(token.ljust(12), style="bold blue")
            token_text.append(" | ")
            token_text.append(bar.ljust(30))
            token_text.append(f" {prob:.4f}\n")

        tokens_panel.update(Panel(token_text, title="Top Candidates", border_style="cyan"))
        gen_panel.update(Panel(Text(step_info['current_text'], style="bold green"), 
                              title="Generated Text", border_style="green"))

        metrics_text = Text()
        metrics_text.append(f"Model: {self.model.model_name}\n")

        metrics_text.append(f"Total Time: {total_time:.2f} sec\n")
        metrics_text.append(f"Avg Token Time: {avg_time_per_token*1000:.1f} ms\n")
        metrics_text.append(f"Strategy: {self.decoding_strategy}\n")
        metrics_text.append(f"Steps: {cur_pos + 1}/{total_steps}\n")
        metrics_panel.update(Panel(metrics_text, title="Metrics 📈", border_style="magenta"))

if __name__ == "__main__":
    # Replace with your model initialization
    class DummyModel:
        async def generate_full_trace(self, prompt, max_steps, temperature, top_p, top_k, decoding_strategy):
            return [{"top_tokens": [{"token": "test", "prob": 0.9}], "current_text": "test"}] * 5

    app = DecodeTUI(model=DummyModel())
    app.run()
