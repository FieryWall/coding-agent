from abc import ABC, abstractmethod
import asyncio
from contextlib import nullcontext
import os
import json
from datetime import datetime
from pathlib import Path
import tiktoken


LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
PRODUCTION_LLM = "gpt-4o-mini"
DEFAULT_TIKTOKEN_ENCODING = "cl100k_base"

# $/1M tokens: (input, output)
MODEL_PRICING = {
    "gpt-4o-mini":  (0.15, 0.60),
    "gpt-4o":       (2.50, 10.00),
    "gpt-4.1":      (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "o4-mini":      (1.10, 4.40),
}

_llm_instance = None


def _get_encoding(model: str = None) -> tiktoken.Encoding:
    if model:
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            pass
    return tiktoken.get_encoding(DEFAULT_TIKTOKEN_ENCODING)


def count_tokens(text: str, model: str = None) -> int:
    return len(_get_encoding(model).encode(text))


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return None
    input_price, output_price = pricing
    return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000


class LLMBackend(ABC):
    truncated: bool = False
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: float = 0.0

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int) -> str:
        pass

    def _get_model_name(self) -> str:
        model = getattr(self, "model", None)
        if isinstance(model, str):
            return model
        return getattr(self, "model_path", None)

    def _log_usage(self, agent_name: str, prompt: str, completion: str, prompt_tokens: int, completion_tokens: int):
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        total = prompt_tokens + completion_tokens
        cumulative = self.total_prompt_tokens + self.total_completion_tokens

        model_name = self._get_model_name()
        cost = estimate_cost(model_name, prompt_tokens, completion_tokens)
        if cost is not None:
            self.total_cost += cost
            print(f"[TOKENS] {agent_name}: prompt={prompt_tokens} completion={completion_tokens} total={total} cost=${cost:.6f} | cumulative={cumulative} total_cost=${self.total_cost:.6f}")
        else:
            print(f"[TOKENS] {agent_name}: prompt={prompt_tokens} completion={completion_tokens} total={total} | cumulative={cumulative}")

        try:
            from langsmith import traceable
            usage_metadata = {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total,
            }
            if cost is not None:
                usage_metadata["total_cost"] = cost

            ls_provider = "openai" if type(self) is OpenAIBackend else "local"

            @traceable(
                run_type="llm",
                name=f"llm_{agent_name}",
                metadata={"ls_provider": ls_provider, "ls_model_name": model_name or "local"},
            )
            def _log_llm_run(messages):
                return {
                    "choices": [{"message": {"content": completion}}],
                    "usage_metadata": usage_metadata,
                }

            _log_llm_run(messages=[{"role": "user", "content": prompt}])
        except Exception:
            pass

    def complete(self, prompt: str, agent_name: str = None, max_tokens: int = 4096) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(LOGS_DIR, exist_ok=True)
        current_max_tokens = max_tokens

        for attempt in range(2):
            cm = nullcontext()
            if agent_name is not None:
                log_file = os.path.join(LOGS_DIR, f"{timestamp}_{agent_name}_attempt{attempt}.log")
                cm = open(log_file, "w", encoding="utf-8")
            try:
                with cm as f:
                    if f:
                        f.write(f"=== {agent_name.upper()} AGENT ===\n")
                        f.write(f"Timestamp: {timestamp}\n")
                        f.write(f"Max tokens: {current_max_tokens}\n")
                        f.write("\n=== PROMPT ===\n")
                        f.write(prompt)
                        f.write("\n\n=== COMPLETION (streaming) ===\n")

                    completion = ""
                    print(f"\n[{agent_name}] ", end="", flush=True)

                    for text in self.generate(prompt, current_max_tokens):
                        completion += text
                        if f: f.write(text); f.flush()

                    print()
                    if f: f.write("\n\n=== END ===\n")

                model_name = self._get_model_name()
                p_tok = count_tokens(prompt, model_name)
                c_tok = count_tokens(completion, model_name)
                self._log_usage(agent_name or "unknown", prompt, completion, p_tok, c_tok)

                if self.truncated and attempt == 0:
                    current_max_tokens *= 2
                    print(f"[WARN] Token limit reached, retrying with {current_max_tokens} tokens...")
                    continue

                return completion
            except Exception as e:
                print(f"\n[LLM ERROR] Completion failed: {e}")
                raise

        return completion

    async def acomplete(self, prompt: str, agent_name: str = None, max_tokens: int = 4096) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(LOGS_DIR, exist_ok=True)
        current_max_tokens = max_tokens
        completion = ""
        for attempt in range(2):
            cm = nullcontext()
            if agent_name is not None:
                log_file = os.path.join(LOGS_DIR, f"{timestamp}_{agent_name}_attempt{attempt}.log")
                cm = open(log_file, "w", encoding="utf-8")
            try:
                if isinstance(self, OpenAIBackend):
                    with cm as f:
                        if f:
                            f.write(f"=== {agent_name.upper()} AGENT ===\n")
                            f.write(f"Timestamp: {timestamp}\n")
                            f.write(f"Max tokens: {current_max_tokens}\n")
                            f.write("\n=== PROMPT ===\n")
                            f.write(prompt)
                            f.write("\n\n=== COMPLETION (streaming) ===\n")
                        completion = ""
                        print(f"\n[{agent_name}] ", end="", flush=True)
                        async for text in self.agenerate(prompt, current_max_tokens):
                            completion += text
                            if f: f.write(text); f.flush()
                        print()
                        if f: f.write("\n\n=== END ===\n")
                else:
                    completion = await asyncio.to_thread(self.complete, prompt, agent_name, current_max_tokens)
                    return completion

                model_name = self._get_model_name()
                p_tok = count_tokens(prompt, model_name)
                c_tok = count_tokens(completion, model_name)
                self._log_usage(agent_name or "unknown", prompt, completion, p_tok, c_tok)

                if self.truncated and attempt == 0:
                    current_max_tokens *= 2
                    print(f"[WARN] Token limit reached, retrying with {current_max_tokens} tokens...")
                    continue
                return completion
            except Exception as e:
                print(f"\n[LLM ERROR] Completion failed: {e}")
                raise
        return completion


OPENAI_MAX_COMPLETION_TOKENS = 16384

class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = PRODUCTION_LLM):
        self.model = model
        self.client = None

    def load(self):
        from openai import OpenAI
        from langsmith.wrappers import wrap_openai
        print(f"[LLM] Using OpenAI API: {self.model}")
        self.client = wrap_openai(OpenAI())
        print("[LLM] OpenAI client initialized")

    def generate(self, prompt: str, max_tokens: int) -> str:
        self.truncated = False
        max_tokens = min(max_tokens, OPENAI_MAX_COMPLETION_TOKENS)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            stream=True,
        )

        completion = ""
        brace_count = 0
        in_json = False
        json_complete = False

        for chunk in response:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            text = choice.delta.content or ""

            if text:
                completion += text
                print(text, end="", flush=True)
                yield text

                if "<|message|>" in completion and not json_complete:
                    after_msg = completion.split("<|message|>")[-1]
                    for c in after_msg:
                        if c == '{':
                            in_json = True
                            brace_count += 1
                        elif c == '}' and in_json:
                            brace_count -= 1
                            if brace_count == 0:
                                json_complete = True
                                break

                    if json_complete:
                        print("\n[STOP] JSON complete")
                        return

            if choice.finish_reason == "length":
                self.truncated = True
            if choice.finish_reason is not None:
                return

    async def agenerate(self, prompt: str, max_tokens: int):
        from openai import AsyncOpenAI
        from langsmith.wrappers import wrap_openai
        self.truncated = False
        max_tokens = min(max_tokens, OPENAI_MAX_COMPLETION_TOKENS)
        client = wrap_openai(AsyncOpenAI())
        stream = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            stream=True,
        )
        completion = ""
        brace_count = 0
        in_json = False
        json_complete = False
        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            text = choice.delta.content or ""
            if text:
                completion += text
                print(text, end="", flush=True)
                yield text
                if "<|message|>" in completion and not json_complete:
                    after_msg = completion.split("<|message|>")[-1]
                    for c in after_msg:
                        if c == '{':
                            in_json = True
                            brace_count += 1
                        elif c == '}' and in_json:
                            brace_count -= 1
                            if brace_count == 0:
                                json_complete = True
                                break
                    if json_complete:
                        print("\n[STOP] JSON complete")
                        return
            if choice.finish_reason == "length":
                self.truncated = True
            if choice.finish_reason is not None:
                return


class LlamaCppBackend(LLMBackend):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load(self):
        from llama_cpp import Llama
        print(f"[LLM] Loading GGUF model from: {self.model_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = Llama(
            model_path=self.model_path,
            n_gpu_layers=1000,
            verbose=False,
            n_ctx=8192,
            flash_attn=True
        )
        print("[LLM] Model loaded successfully (flash-attn enabled)")

    def generate(self, prompt: str, max_tokens: int) -> str:
        self.truncated = False
        response = self.model.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repeat_penalty=1.1,
            stream=True,
            stop=["[DONE]", "\n\nThinking:"]
        )

        completion = ""
        brace_count = 0
        in_json = False
        json_complete = False

        for token in response:
            if "choices" not in token or not token["choices"]:
                continue

            choice = token["choices"][0]
            text = choice.get("text", "")

            if text:
                completion += text
                print(text, end="", flush=True)
                yield text

                if "[DONE]" in completion:
                    print("\n[STOP] [DONE] token")
                    return
                if "<|message|>" in completion and not json_complete:
                    after_msg = completion.split("<|message|>")[-1]
                    for c in after_msg:
                        if c == '{':
                            in_json = True
                            brace_count += 1
                        elif c == '}' and in_json:
                            brace_count -= 1
                            if brace_count == 0:
                                json_complete = True
                                break

                    if json_complete:
                        print("\n[STOP] JSON complete")
                        return

            finish_reason = choice.get("finish_reason")
            if finish_reason == "length":
                self.truncated = True
            if finish_reason is not None:
                return


class MLXBackend(LLMBackend):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    def load(self):
        from mlx_lm import load
        print(f"[LLM] Loading MLX model from: {self.model_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model, self.tokenizer = load(self.model_path)
        print("[LLM] Model loaded successfully")

    def generate(self, prompt: str, max_tokens: int) -> str:
        from mlx_lm import stream_generate

        self.truncated = False
        completion = ""
        brace_count = 0
        in_json = False
        json_complete = False
        token_count = 0

        for token_data in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens
        ):
            token_count += 1
            text = token_data.text if hasattr(token_data, 'text') else str(token_data)

            if text:
                completion += text
                print(text, end="", flush=True)
                yield text

                if "[DONE]" in completion:
                    print("\n[STOP] [DONE] token")
                    return
                if "<|message|>" in completion and not json_complete:
                    after_msg = completion.split("<|message|>")[-1]
                    for c in after_msg:
                        if c == '{':
                            in_json = True
                            brace_count += 1
                        elif c == '}' and in_json:
                            brace_count -= 1
                            if brace_count == 0:
                                json_complete = True
                                break

                    if json_complete:
                        print("\n[STOP] JSON complete")
                        return

        if token_count >= max_tokens:
            self.truncated = True


class RemoteBackend(OpenAIBackend):
    """Connects to any OpenAI-compatible HTTP server (llama.cpp server, Ollama, vLLM, etc.)."""

    def __init__(self, base_url: str, model: str = "local", api_key: str = "none"):
        super().__init__(model)
        self.base_url = base_url
        self.api_key = api_key

    def load(self):
        from openai import OpenAI
        print(f"[LLM] Connecting to remote server: {self.base_url} (model: {self.model})")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        print("[LLM] Remote client initialized")

    async def agenerate(self, prompt: str, max_tokens: int):
        from openai import AsyncOpenAI
        self.truncated = False
        max_tokens = min(max_tokens, OPENAI_MAX_COMPLETION_TOKENS)
        client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        stream = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            stream=True,
        )
        completion = ""
        brace_count = 0
        in_json = False
        json_complete = False
        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            text = choice.delta.content or ""
            if text:
                completion += text
                print(text, end="", flush=True)
                yield text
                if "<|message|>" in completion and not json_complete:
                    after_msg = completion.split("<|message|>")[-1]
                    for c in after_msg:
                        if c == '{':
                            in_json = True
                            brace_count += 1
                        elif c == '}' and in_json:
                            brace_count -= 1
                            if brace_count == 0:
                                json_complete = True
                                break
                    if json_complete:
                        print("\n[STOP] JSON complete")
                        return
            if choice.finish_reason == "length":
                self.truncated = True
            if choice.finish_reason is not None:
                return


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def detect_backend(model_path: str, remote_model: str = "local") -> LLMBackend:
    if _is_url(model_path):
        return RemoteBackend(model_path, model=remote_model)

    path = Path(model_path)

    if path.suffix == ".gguf":
        return LlamaCppBackend(model_path)

    if path.is_dir():
        return MLXBackend(model_path)

    safetensors = list(path.parent.glob("*.safetensors")) if path.parent.exists() else []
    if safetensors or path.suffix == ".safetensors":
        return MLXBackend(str(path.parent) if path.suffix == ".safetensors" else model_path)

    return LlamaCppBackend(model_path)


def create_llm_client(model_path: str = None, model_name: str = PRODUCTION_LLM, remote_model: str = "local") -> LLMBackend:
    if model_path is None:
        llm_instance = OpenAIBackend(model_name)
    else:
        llm_instance = detect_backend(model_path, remote_model)
    llm_instance.load()
    return llm_instance


def extract_json(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def extract_json_array(text: str) -> list | None:
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "[":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            return obj
    return None
