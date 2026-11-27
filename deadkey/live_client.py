"""Universal Local OpenAI-Compatible Client (TensorFold v1.3.5, vLLM, SGLang, ExLlamaV3, llama-server).

Features:
- Zero heavy dependencies (pure stdlib HTTP keep-alive connection pooling + thread-safe locking).
- Real SSE token-by-token streaming (`chat_stream`) with usage/timings/reasoning extraction.
- Automatic `/props` & `/v1/models` discovery (Switchyard-style `context_discovery`, closes Defect #9).
- Multi-slot prefix isolation (`id_slot=0` live chat, `id_slot=1` cognition/sleep-time).
- Dual tool protocol: Native OpenAI `tools` OR `InklingBypassAdapter` (for llama.cpp `inkling2_loop`).
- First-token logprob extraction (`logprobs=True, top_logprobs=5`) for `VGR-Local`.
- Exact telemetry: `ttft_ms`, `total_ms`, `tok_s`, `prompt_tokens`, `cache_n`, `kv_hit_ratio`.
"""
from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from .providers import InklingBypassAdapter, ParsedModelTurn


@dataclass
class BackendInfo:
    base_url: str
    reachable: bool
    model_id: str = "unknown"
    ctx_size: int = 32768
    total_slots: int = 1
    engine_guess: str = "unknown"
    error: str = ""


@dataclass
class LiveTurnTelemetry:
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_n: Optional[int] = None
    kv_hit_ratio: Optional[float] = None
    tok_s: float = 0.0
    first_token_logprob: Optional[float] = None
    top_logprob_margin: Optional[float] = None
    raw_finish_reason: str = ""


class LocalLLMClient:
    """Thread-safe HTTP keep-alive & streaming client for any local OpenAI-compatible engine."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11455",
        model: str = "auto",
        api_key: str = "local",
        tool_protocol: str = "openai_native",
        disable_thinking: bool = True,
        default_timeout: float = 120.0,
    ):
        self.model = model
        self.api_key = api_key
        self.tool_protocol = tool_protocol
        self.disable_thinking = disable_thinking
        self.default_timeout = default_timeout
        self._lock = threading.RLock()
        self._conn: Optional[http.client.HTTPConnection] = None
        self.set_endpoint(base_url, tool_protocol=tool_protocol, disable_thinking=disable_thinking)

    def set_endpoint(
        self,
        base_url: str,
        *,
        tool_protocol: Optional[str] = None,
        disable_thinking: Optional[bool] = None,
    ) -> None:
        with self._lock:
            self.base_url = base_url.rstrip("/")
            parsed = urllib.parse.urlparse(self.base_url)
            self._scheme = parsed.scheme or "http"
            self._host = parsed.hostname or "127.0.0.1"
            self._port = parsed.port or (443 if self._scheme == "https" else 80)
            self._prefix = parsed.path.rstrip("/")
            if tool_protocol is not None:
                self.tool_protocol = tool_protocol
            if disable_thinking is not None:
                self.disable_thinking = disable_thinking
            self._reset_conn()

    def _new_conn(self, timeout: float) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self._scheme == "https" else http.client.HTTPConnection
        return cls(self._host, self._port, timeout=timeout)

    def _get_conn(self, timeout: float) -> http.client.HTTPConnection:
        if self._conn is not None:
            self._conn.timeout = timeout
            return self._conn
        self._conn = self._new_conn(timeout)
        return self._conn

    def _reset_conn(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _request_json(self, method: str, path: str, body: Optional[dict] = None,
                      timeout: Optional[float] = None) -> tuple[int, dict[str, Any]]:
        t = timeout or self.default_timeout
        headers = {"Authorization": f"Bearer {self.api_key}", "Connection": "keep-alive"}
        payload = None
        if body is not None:
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        full_path = f"{self._prefix}{path}"
        with self._lock:
            for attempt in range(2):
                try:
                    conn = self._get_conn(t)
                    conn.request(method, full_path, body=payload, headers=headers)
                    resp = conn.getresponse()
                    raw = resp.read().decode("utf-8", errors="replace")
                    data = json.loads(raw) if raw.strip().startswith("{") else {"raw": raw}
                    return resp.status, data
                except (http.client.HTTPException, OSError) as exc:
                    self._reset_conn()
                    if attempt == 1:
                        raise RuntimeError(f"Backend {self.base_url}{path} unreachable: {exc}") from exc
        raise RuntimeError("Unreachable")

    def discover(self, timeout: float = 1.5) -> BackendInfo:
        """Auto-discovers live model name, ctx, slots via `/props` and `/v1/models` (closes Defect #9)."""
        info = BackendInfo(base_url=self.base_url, reachable=False)
        try:
            code, props = self._request_json("GET", "/props", timeout=timeout)
            if code == 200 and isinstance(props, dict):
                info.reachable = True
                info.engine_guess = "tensorfold" if "tensorfold" in json.dumps(props).lower() else "llama_server"
                info.total_slots = int(props.get("total_slots", 1) or 1)
                dgs = props.get("default_generation_settings") or {}
                info.ctx_size = int(dgs.get("n_ctx") or props.get("n_ctx") or 32768)
                mp = props.get("model_path") or props.get("model_alias") or ""
                if mp:
                    info.model_id = str(mp).replace("\\", "/").split("/")[-1]
        except Exception as exc:
            info.error = str(exc)

        try:
            code, data = self._request_json("GET", "/v1/models", timeout=timeout)
            if code == 200 and isinstance(data, dict):
                info.reachable = True
                models = data.get("data") or []
                if models and isinstance(models, list):
                    mid = models[0].get("id")
                    if mid:
                        info.model_id = str(mid)
                    max_len = models[0].get("max_model_len")
                    if max_len:
                        info.ctx_size = int(max_len)
                if info.engine_guess == "unknown":
                    info.engine_guess = "openai_compatible"
                info.error = ""
        except Exception as exc:
            if not info.reachable:
                info.error = str(exc)
        return info

    def _build_body(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        slot_id: int,
        max_tokens: int,
        temperature: float,
        request_logprobs: bool,
        disable_thinking: Optional[bool],
        model_override: Optional[str],
        stream: bool,
    ) -> dict[str, Any]:
        model_name = model_override or self.model
        if model_name == "auto":
            disc = self.discover(timeout=1.5)
            model_name = disc.model_id if disc.reachable else "local-model"

        full_msgs = [{"role": "system", "content": system_prompt}] + list(messages)
        body: dict[str, Any] = {
            "model": model_name,
            "messages": full_msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            "id_slot": slot_id,
            "cache_prompt": True,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        dt = self.disable_thinking if disable_thinking is None else disable_thinking
        if dt:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        has_native_tools = bool(tools and self.tool_protocol == "openai_native")
        # llama-server rejects `logprobs=True` when BOTH `stream=True` and `tools` are active!
        if request_logprobs and not (stream and has_native_tools):
            body["logprobs"] = True
            body["top_logprobs"] = 5
        if has_native_tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        return body

    @staticmethod
    def _parse_tool_calls(raw_tcs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parsed_calls: list[dict[str, Any]] = []
        for tc in raw_tcs:
            fn = tc.get("function") or {}
            name = (fn.get("name") or "").strip()
            args_raw = fn.get("arguments")
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw) if args_raw.strip() else {}
                except json.JSONDecodeError:
                    args = {}
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                args = {}
            if name:
                parsed_calls.append({"id": tc.get("id"), "name": name, "arguments": args})
        return parsed_calls

    def chat_once(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        slot_id: int = 0,
        max_tokens: int = 600,
        temperature: float = 0.6,
        request_logprobs: bool = True,
        disable_thinking: Optional[bool] = None,
        model_override: Optional[str] = None,
    ) -> tuple[ParsedModelTurn, LiveTurnTelemetry]:
        """Executes one non-streaming chat completion with slot isolation, logprobs, and exact timings."""
        body = self._build_body(
            system_prompt=system_prompt, messages=messages, tools=tools, slot_id=slot_id,
            max_tokens=max_tokens, temperature=temperature, request_logprobs=request_logprobs,
            disable_thinking=disable_thinking, model_override=model_override, stream=False,
        )
        t0 = time.perf_counter()
        status, data = self._request_json("POST", "/v1/chat/completions", body=body)
        if status == 400 and body.get("logprobs") and "logprobs" in json.dumps(data).lower():
            body.pop("logprobs", None)
            body.pop("top_logprobs", None)
            t0 = time.perf_counter()
            status, data = self._request_json("POST", "/v1/chat/completions", body=body)
        total_ms = (time.perf_counter() - t0) * 1000.0
        if status != 200:
            raise RuntimeError(f"HTTP {status} from {self.base_url}: {json.dumps(data)[:300]}")

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = (msg.get("content") or "").strip()
        parsed_calls = self._parse_tool_calls(msg.get("tool_calls") or [])

        if self.tool_protocol == "xml_json_bypass" and not parsed_calls and "<tool_call>" in content:
            bp = InklingBypassAdapter.parse_response(content)
            content = bp.text
            parsed_calls = bp.tool_calls

        usage = data.get("usage") or {}
        timings = data.get("timings") or {}
        p_toks = int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0)
        c_toks = int(usage.get("completion_tokens") or timings.get("predicted_n") or 0)
        cache_n = timings.get("cache_n")
        if cache_n is None:
            cache_n = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        kv_hit = None
        if cache_n is not None and (p_toks + int(cache_n)) > 0:
            denom = p_toks if p_toks >= int(cache_n) else (p_toks + int(cache_n))
            kv_hit = round(min(1.0, int(cache_n) / max(1, denom)), 3)

        ttft_ms = float(timings.get("prompt_ms") or total_ms)
        pred_ms = float(timings.get("predicted_ms") or 0.0)
        tok_s = round(c_toks / (pred_ms / 1000.0), 1) if pred_ms > 0 else (
            round(c_toks / max(0.001, total_ms / 1000.0), 1) if c_toks else 0.0
        )

        first_lp = None
        margin_lp = None
        lp_content = ((choice.get("logprobs") or {}).get("content") or [])
        if lp_content and isinstance(lp_content, list):
            tok0 = lp_content[0] or {}
            if tok0.get("logprob") is not None:
                first_lp = float(tok0["logprob"])
            tops = tok0.get("top_logprobs") or []
            if len(tops) >= 2 and tops[0].get("logprob") is not None and tops[1].get("logprob") is not None:
                margin_lp = round(float(tops[0]["logprob"]) - float(tops[1]["logprob"]), 4)

        tel = LiveTurnTelemetry(
            ttft_ms=round(ttft_ms, 1),
            total_ms=round(total_ms, 1),
            prompt_tokens=p_toks,
            completion_tokens=c_toks,
            cache_n=int(cache_n) if cache_n is not None else None,
            kv_hit_ratio=kv_hit,
            tok_s=tok_s,
            first_token_logprob=first_lp,
            top_logprob_margin=margin_lp,
            raw_finish_reason=str(choice.get("finish_reason") or ""),
        )
        turn = ParsedModelTurn(
            text=content,
            tool_calls=parsed_calls,
            ttft_ms=tel.ttft_ms,
            total_ms=tel.total_ms,
            cache_hit_ratio=tel.kv_hit_ratio,
            protocol_used=self.tool_protocol,
        )
        return turn, tel

    def chat_stream(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        slot_id: int = 0,
        max_tokens: int = 800,
        temperature: float = 0.6,
        request_logprobs: bool = True,
        disable_thinking: Optional[bool] = None,
        model_override: Optional[str] = None,
    ) -> Iterator[dict[str, Any]]:
        """Streams real token-by-token chunks from OpenAI-compatible SSE `/v1/chat/completions`.

        Yields dicts:
          - `{"type": "delta", "content": "..."}`
          - `{"type": "thinking", "content": "..."}`
          - `{"type": "final", "turn": ParsedModelTurn, "telemetry": LiveTurnTelemetry, "thinking": "..."}`
        """
        body = self._build_body(
            system_prompt=system_prompt, messages=messages, tools=tools, slot_id=slot_id,
            max_tokens=max_tokens, temperature=temperature, request_logprobs=request_logprobs,
            disable_thinking=disable_thinking, model_override=model_override, stream=True,
        )
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Connection": "close",
        }
        full_path = f"{self._prefix}/v1/chat/completions"
        conn = self._new_conn(self.default_timeout)
        t0 = time.perf_counter()
        first_tok_ms: Optional[float] = None
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tc_accum: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        timings: dict[str, Any] = {}
        finish_reason = ""
        first_lp: Optional[float] = None
        margin_lp: Optional[float] = None

        try:
            conn.request("POST", full_path, body=payload, headers=headers)
            resp = conn.getresponse()
            if resp.status == 400 and body.get("logprobs"):
                err_raw = resp.read().decode("utf-8", errors="replace")[:300]
                if "logprobs" in err_raw.lower():
                    try:
                        conn.close()
                    except Exception:
                        pass
                    body.pop("logprobs", None)
                    body.pop("top_logprobs", None)
                    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
                    conn = self._new_conn(self.default_timeout)
                    t0 = time.perf_counter()
                    conn.request("POST", full_path, body=payload, headers=headers)
                    resp = conn.getresponse()
                else:
                    raise RuntimeError(f"HTTP 400 from {self.base_url}: {err_raw}")
            if resp.status != 200:
                err_raw = resp.read().decode("utf-8", errors="replace")[:300]
                raise RuntimeError(f"HTTP {resp.status} from {self.base_url}: {err_raw}")

            while True:
                raw_line = resp.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if chunk.get("usage"):
                    usage = chunk["usage"]
                if chunk.get("timings"):
                    timings = chunk["timings"]

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                ch0 = choices[0]
                if ch0.get("finish_reason"):
                    finish_reason = str(ch0["finish_reason"])

                if first_lp is None and ch0.get("logprobs"):
                    lpc = (ch0["logprobs"] or {}).get("content") or []
                    if lpc and isinstance(lpc, list):
                        tok0 = lpc[0] or {}
                        if tok0.get("logprob") is not None:
                            first_lp = float(tok0["logprob"])
                        tops = tok0.get("top_logprobs") or []
                        if len(tops) >= 2 and tops[0].get("logprob") is not None and tops[1].get("logprob") is not None:
                            margin_lp = round(float(tops[0]["logprob"]) - float(tops[1]["logprob"]), 4)

                delta = ch0.get("delta") or {}
                r_tok = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if r_tok:
                    if first_tok_ms is None:
                        first_tok_ms = (time.perf_counter() - t0) * 1000.0
                    thinking_parts.append(r_tok)
                    yield {"type": "thinking", "content": r_tok}

                c_tok = delta.get("content") or ""
                if c_tok:
                    if first_tok_ms is None:
                        first_tok_ms = (time.perf_counter() - t0) * 1000.0
                    content_parts.append(c_tok)
                    yield {"type": "delta", "content": c_tok}

                for tc in (delta.get("tool_calls") or []):
                    idx = int(tc.get("index", 0))
                    slot = tc_accum.setdefault(idx, {"id": tc.get("id"), "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]
        finally:
            try:
                conn.close()
            except Exception:
                pass

        total_ms = (time.perf_counter() - t0) * 1000.0
        content = "".join(content_parts).strip()
        thinking = "".join(thinking_parts).strip()
        raw_tcs = [tc_accum[k] for k in sorted(tc_accum.keys())]
        parsed_calls = self._parse_tool_calls(raw_tcs)

        if self.tool_protocol == "xml_json_bypass" and not parsed_calls and "<tool_call>" in content:
            bp = InklingBypassAdapter.parse_response(content)
            content = bp.text
            parsed_calls = bp.tool_calls

        p_toks = int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0)
        c_toks = int(usage.get("completion_tokens") or timings.get("predicted_n") or max(1, len(content) // 4))
        cache_n = timings.get("cache_n")
        if cache_n is None:
            cache_n = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        kv_hit = None
        if cache_n is not None and (p_toks + int(cache_n)) > 0:
            denom = p_toks if p_toks >= int(cache_n) else (p_toks + int(cache_n))
            kv_hit = round(min(1.0, int(cache_n) / max(1, denom)), 3)

        ttft_ms = float(timings.get("prompt_ms") or first_tok_ms or total_ms)
        pred_ms = float(timings.get("predicted_ms") or max(1.0, total_ms - ttft_ms))
        tok_s = round(c_toks / (pred_ms / 1000.0), 1) if pred_ms > 0 and c_toks else 0.0

        tel = LiveTurnTelemetry(
            ttft_ms=round(ttft_ms, 1),
            total_ms=round(total_ms, 1),
            prompt_tokens=p_toks,
            completion_tokens=c_toks,
            cache_n=int(cache_n) if cache_n is not None else None,
            kv_hit_ratio=kv_hit,
            tok_s=tok_s,
            first_token_logprob=first_lp,
            top_logprob_margin=margin_lp,
            raw_finish_reason=finish_reason,
        )
        turn = ParsedModelTurn(
            text=content,
            tool_calls=parsed_calls,
            ttft_ms=tel.ttft_ms,
            total_ms=tel.total_ms,
            cache_hit_ratio=tel.kv_hit_ratio,
            protocol_used=self.tool_protocol,
        )
        yield {"type": "final", "turn": turn, "telemetry": tel, "thinking": thinking}
