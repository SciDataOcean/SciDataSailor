import asyncio
import re
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI


class LLMClient:
    """OpenAI-compatible API client pool (adapted from ARPO VLLMClientPool)."""

    def __init__(
        self,
        endpoints: List[str],
        api_keys: Optional[List[str]] = None,
        default_model: str = "gpt-4",
    ):
        self.clients: List[AsyncOpenAI] = []
        api_keys = api_keys or ["EMPTY"] * len(endpoints)
        if len(api_keys) != len(endpoints):
            raise ValueError("len(api_keys) != len(endpoints)")
        for endpoint, api_key in zip(endpoints, api_keys):
            self.clients.append(
                AsyncOpenAI(base_url=endpoint, api_key=api_key)
            )
        self.default_model = default_model
        self.current_client_idx = 0
        self.lock = asyncio.Lock()
        self.session_to_client: Dict[str, int] = {}
        print(f"Initialized LLM client pool with {len(endpoints)} endpoint(s)")

    async def _get_client(self, session_id: Optional[str] = None) -> AsyncOpenAI:
        async with self.lock:
            if not session_id:
                client = self.clients[self.current_client_idx]
                self.current_client_idx = (self.current_client_idx + 1) % len(self.clients)
                return client
            if session_id in self.session_to_client:
                return self.clients[self.session_to_client[session_id]]
            idx = self.current_client_idx
            self.session_to_client[session_id] = idx
            self.current_client_idx = (self.current_client_idx + 1) % len(self.clients)
            return self.clients[idx]

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.6,
        max_tokens: int = 8192,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        client = await self._get_client(session_id)
        current_max_tokens = max(64, int(max_tokens))
        for attempt in range(3):
            try:
                request_kwargs: Dict[str, Any] = {
                    "model": self.default_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": current_max_tokens,
                    "top_p": top_p,
                }
                if stop is not None:
                    request_kwargs["stop"] = stop
                if logprobs:
                    request_kwargs["logprobs"] = True
                    if top_logprobs is not None:
                        request_kwargs["top_logprobs"] = int(top_logprobs)
                response = await client.chat.completions.create(**request_kwargs)
                return {
                    "text": self._extract_with_thinking(response),
                    "usage": self._extract_usage(response),
                    "logprobs": self._extract_logprobs(response) if logprobs else None,
                }
            except Exception as e:
                if self._is_unsupported_logprobs_error(e) and logprobs:
                    print(f"[Warning] 'logprobs' not supported, retrying without it")
                    try:
                        fallback_kwargs: Dict[str, Any] = {
                            "model": self.default_model,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": current_max_tokens,
                            "top_p": top_p,
                        }
                        if stop is not None:
                            fallback_kwargs["stop"] = stop
                        response = await client.chat.completions.create(**fallback_kwargs)
                        return {
                            "text": self._extract_with_thinking(response),
                            "usage": self._extract_usage(response),
                            "logprobs": None,
                        }
                    except Exception as e2:
                        print(f"Fallback (no logprobs) also failed: {e2}")
                if self._is_unsupported_stop_error(e) and stop is not None:
                    print(f"[Warning] 'stop' not supported, retrying without it")
                    try:
                        fallback_kwargs: Dict[str, Any] = {
                            "model": self.default_model,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": current_max_tokens,
                            "top_p": top_p,
                        }
                        if logprobs:
                            fallback_kwargs["logprobs"] = True
                            if top_logprobs is not None:
                                fallback_kwargs["top_logprobs"] = int(top_logprobs)
                        response = await client.chat.completions.create(**fallback_kwargs)
                        return {
                            "text": self._extract_with_thinking(response),
                            "usage": self._extract_usage(response),
                            "logprobs": self._extract_logprobs(response) if logprobs else None,
                        }
                    except Exception as e2:
                        print(f"Fallback also failed: {e2}")
                if self._is_context_overflow_error(e):
                    reduced = self._reduce_max_tokens_for_overflow(
                        e, current_max_tokens
                    )
                    if reduced < current_max_tokens:
                        print(
                            f"[Warning] context overflow, reducing max_tokens: "
                            f"{current_max_tokens} -> {reduced}"
                        )
                        current_max_tokens = reduced
                        if attempt < 2:
                            await asyncio.sleep(0.3 * (attempt + 1))
                        continue
                print(f"LLM chat_completion failed (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    await asyncio.sleep(1 * (attempt + 1))
        return await self._retry_chat_with_other_client(
            messages, temperature, current_max_tokens, top_p, stop, session_id,
            logprobs=logprobs, top_logprobs=top_logprobs,
        )

    async def _retry_chat_with_other_client(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop: Optional[List[str]],
        session_id: Optional[str],
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        original_idx = self.session_to_client.get(session_id, self.current_client_idx)
        tried = {original_idx}
        while len(tried) < len(self.clients):
            async with self.lock:
                next_idx = (original_idx + 1) % len(self.clients)
                while next_idx in tried:
                    next_idx = (next_idx + 1) % len(self.clients)
                tried.add(next_idx)
                if session_id:
                    self.session_to_client[session_id] = next_idx
            client = self.clients[next_idx]
            try:
                request_kwargs: Dict[str, Any] = {
                    "model": self.default_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                }
                if stop is not None:
                    request_kwargs["stop"] = stop
                if logprobs:
                    request_kwargs["logprobs"] = True
                    if top_logprobs is not None:
                        request_kwargs["top_logprobs"] = int(top_logprobs)
                response = await client.chat.completions.create(**request_kwargs)
                return {
                    "text": self._extract_with_thinking(response),
                    "usage": self._extract_usage(response),
                    "logprobs": self._extract_logprobs(response) if logprobs else None,
                }
            except Exception as e:
                print(f"Retry chat_completion with client {next_idx} failed: {e}")
        print("All LLM clients failed, returning None")
        return None

    @staticmethod
    def _extract_with_thinking(response) -> str:
        """Extract internal thinking (if any) and wrap it with <think> tags before the content."""
        msg = response.choices[0].message
        content = msg.content or ""
        thinking = getattr(msg, "reasoning_content", None) or getattr(msg, "thinking", None)
        if thinking:
            return f"<think>{thinking}</think> {content}"
        return content

    @staticmethod
    def _extract_usage(response) -> Dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }

    @staticmethod
    def _is_unsupported_stop_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return ("unsupported parameter" in text and "'stop'" in text) or \
               ("stop is not supported" in text)

    @staticmethod
    def _is_unsupported_logprobs_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return ("logprobs" in text and (
            "unsupported" in text or "not supported" in text
            or "unknown" in text or "invalid" in text
        ))

    @staticmethod
    def _extract_logprobs(response) -> Optional[List[Dict[str, Any]]]:
        if not response.choices:
            return None
        choice = response.choices[0]
        lp_container = choice.logprobs
        if lp_container is None:
            return None
        content = lp_container.content
        if not content:
            return None

        out: List[Dict[str, Any]] = []
        for tok in content:
            top_list = [
                {
                    "token": t.token,
                    "logprob": float(t.logprob or 0.0),
                }
                for t in (tok.top_logprobs or [])
            ]
            out.append({
                "token": tok.token,
                "logprob": float(tok.logprob or 0.0),
                "top_logprobs": top_list,
            })
        return out

    @staticmethod
    def _is_context_overflow_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            "maximum context length" in text
            or "context length" in text
            or "input_tokens" in text
            or "too many tokens" in text
        )

    @staticmethod
    def _reduce_max_tokens_for_overflow(exc: Exception, current_max_tokens: int) -> int:
        msg = str(exc)
        max_ctx = None
        in_tokens = None

        m_ctx = re.search(r"maximum context length is (\d+)", msg, flags=re.IGNORECASE)
        if m_ctx:
            max_ctx = int(m_ctx.group(1))
        m_in = re.search(
            r"contains at least (\d+) input tokens",
            msg,
            flags=re.IGNORECASE,
        )
        if m_in:
            in_tokens = int(m_in.group(1))

        # Keep a small safety margin for provider-side token accounting variance.
        if max_ctx is not None and in_tokens is not None:
            available = max(32, max_ctx - in_tokens - 64)
            return max(32, min(current_max_tokens // 2, available))

        return max(32, current_max_tokens // 2)

    async def text_completion(
        self,
        prompt: str,
        temperature: float = 0.6,
        max_tokens: int = 8192,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        top_k: int = 20,
        min_p: float = 0.0,
        repetition_penalty: float = 1.1,
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Send a text completion request and return {'text', 'usage'}."""
        client = await self._get_client(session_id)
        for attempt in range(3):
            try:
                request_kwargs: Dict[str, Any] = {
                    "model": self.default_model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "extra_body": {
                        "top_k": top_k,
                        "min_p": min_p,
                        "repetition_penalty": repetition_penalty,
                        "include_stop_str_in_output": True,
                    },
                }
                if stop is not None:
                    request_kwargs["stop"] = stop
                response = await client.completions.create(**request_kwargs)
                return {
                    "text": response.choices[0].text if response.choices else "",
                    "usage": self._extract_usage(response),
                }
            except Exception as e:
                if self._is_extra_body_error(e) and attempt == 0:
                    print(f"[Warning] extra_body not supported, retrying without it")
                    try:
                        fallback_kwargs: Dict[str, Any] = {
                            "model": self.default_model,
                            "prompt": prompt,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "top_p": top_p,
                        }
                        if stop is not None:
                            fallback_kwargs["stop"] = stop
                        response = await client.completions.create(**fallback_kwargs)
                        return {
                            "text": response.choices[0].text if response.choices else "",
                            "usage": self._extract_usage(response),
                        }
                    except Exception as e2:
                        print(f"Fallback also failed: {e2}")
                print(f"LLM text_completion failed (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    await asyncio.sleep(1 * (attempt + 1))
        return await self._retry_text_with_other_client(
            prompt, temperature, max_tokens, top_p, stop,
            top_k, min_p, repetition_penalty, session_id,
        )

    async def _retry_text_with_other_client(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop: Optional[List[str]],
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        session_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        original_idx = self.session_to_client.get(session_id, self.current_client_idx)
        tried = {original_idx}
        while len(tried) < len(self.clients):
            async with self.lock:
                next_idx = (original_idx + 1) % len(self.clients)
                while next_idx in tried:
                    next_idx = (next_idx + 1) % len(self.clients)
                tried.add(next_idx)
                if session_id:
                    self.session_to_client[session_id] = next_idx
            client = self.clients[next_idx]
            try:
                request_kwargs: Dict[str, Any] = {
                    "model": self.default_model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "extra_body": {
                        "top_k": top_k,
                        "min_p": min_p,
                        "repetition_penalty": repetition_penalty,
                        "include_stop_str_in_output": True,
                    },
                }
                if stop is not None:
                    request_kwargs["stop"] = stop
                response = await client.completions.create(**request_kwargs)
                return {
                    "text": response.choices[0].text if response.choices else "",
                    "usage": self._extract_usage(response),
                }
            except Exception as e:
                print(f"Retry text_completion with client {next_idx} failed: {e}")
        print("All LLM clients failed, returning None")
        return None

    @staticmethod
    def _is_extra_body_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "extra_body" in text or "unrecognized request argument" in text
