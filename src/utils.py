"""
Provider-specific API key validation utilities.
"""

import json
from urllib import error, request
from urllib.parse import quote

import rich
from openai import APIStatusError, AuthenticationError, OpenAI, RateLimitError

from configs import PROVIDER_PATTERNS

DEFAULT_TIMEOUT = 20

# OpenAI-compatible probes aligned with OmniRoute providerRegistry base URLs / first model id.
_OPENAI_COMPAT_PROBES: dict[str, tuple[str, str, dict[str, str] | None]] = {
    "openrouter": (
        "openai/gpt-4o-mini",
        "https://openrouter.ai/api/v1",
        {"HTTP-Referer": "https://endpoint-proxy.local", "X-Title": "Endpoint Proxy"},
    ),
    "nvidia": ("minimaxai/minimax-m2.7", "https://integrate.api.nvidia.com/v1", None),
    "groq": ("llama-3.1-8b-instant", "https://api.groq.com/openai/v1", None),
    "deepseek": ("deepseek-v4-flash", "https://api.deepseek.com/v1", None),
    "xai": ("grok-4.3", "https://api.x.ai/v1", None),
    "mistral": ("mistral-small-latest", "https://api.mistral.ai/v1", None),
    "together": (
        "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "https://api.together.xyz/v1",
        None,
    ),
    "fireworks": (
        "accounts/fireworks/models/kimi-k2p6",
        "https://api.fireworks.ai/inference/v1",
        None,
    ),
    "cerebras": ("zai-glm-4.7", "https://api.cerebras.ai/v1", None),
    "nebius": ("meta-llama/Llama-3.3-70B-Instruct", "https://api.tokenfactory.nebius.com/v1", None),
    "moonshot": ("kimi-k2.6", "https://api.moonshot.ai/v1", None),
    "agentrouter": (
        "auto",
        "https://agentrouter.org/v1",
        {"HTTP-Referer": "https://endpoint-proxy.local", "X-Title": "OmniRoute"},
    ),
    "siliconflow": ("deepseek-ai/DeepSeek-V3.2", "https://api.siliconflow.com/v1", None),
    "qianfan": ("ernie-4.5-turbo-128k", "https://qianfan.baidubce.com/v2", None),
    "ollama-cloud": ("deepseek-v4-flash", "https://ollama.com/v1", None),
    "blackbox": ("gpt-4o", "https://api.blackbox.ai/v1", None),
    "meta-llama": ("Llama-3.3-70B-Instruct", "https://api.llama.com/compat/v1", None),
    "publicai": ("swiss-ai/apertus-70b-instruct", "https://api.publicai.co/v1", None),
    "hyperbolic": ("Qwen/QwQ-32B", "https://api.hyperbolic.xyz/v1", None),
    # OmniRoute registry — OpenAI-compatible chat completions
    "aimlapi": ("gpt-4o", "https://api.aimlapi.com/v1", None),
    "alibaba": ("qwen-max", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", None),
    "alicode": ("qwen3.5-plus", "https://coding.dashscope.aliyuncs.com/v1", None),
    "alicode-intl": ("qwen3.5-plus", "https://coding-intl.dashscope.aliyuncs.com/v1", None),
    "baseten": ("moonshotai/Kimi-K2.5", "https://inference.baseten.co/v1", None),
    "codestral": ("codestral-latest", "https://codestral.mistral.ai/v1", None),
    "crof": ("deepseek-v4-pro", "https://api.crof.ai/v1", None),
    "deepinfra": ("Qwen/Qwen3-Coder-480B-A35B-Instruct", "https://api.deepinfra.com/v1/openai", None),
    "featherless-ai": ("featherless-ai/Qwerky-72B", "https://api.featherless.ai/v1", None),
    "friendliai": ("meta-llama-3.1-70b-instruct", "https://api.friendli.ai/dedicated/v1", None),
    "galadriel": ("galadriel-latest", "https://api.galadriel.ai/v1", None),
    "glm-cn": ("glm-5.1", "https://open.bigmodel.cn/api/coding/paas/v4", None),
    "heroku": ("claude-3-5-sonnet-latest", "https://us.inference.heroku.com/v1", None),
    "inference-net": ("meta-llama/Llama-3.3-70B-Instruct", "https://api.inference.net/v1", None),
    "kimi": ("kimi-k2.6", "https://api.moonshot.ai/v1", None),
    "lambda-ai": ("deepseek-r1-671b", "https://api.lambda.ai/v1", None),
    "llamagate": ("qwen2.5-coder-7b", "https://llamagate.ai/v1", None),
    "longcat": ("LongCat-Flash-Lite", "https://api.longcat.chat/openai/v1", None),
    "morph": ("morph-v3-fast", "https://api.morphllm.com/v1", None),
    "nanogpt": ("gpt-4o-mini", "https://nano-gpt.com/api/v1", None),
    "nous-research": ("Hermes-4-405B", "https://inference-api.nousresearch.com/v1", None),
    "nscale": ("Qwen/QwQ-32B", "https://inference.api.nscale.com/v1", None),
    "opencode-go": ("glm-5.1", "https://opencode.ai/zen/v1", None),
    "opencode-zen": ("gpt-5-nano", "https://opencode.ai/zen/v1", None),
    "ovhcloud": ("Meta-Llama-3_3-70B-Instruct", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1", None),
    "pollinations": ("openai", "https://text.pollinations.ai/openai", None),
    "predibase": ("llama-3.3-70b", "https://serving.app.predibase.com/v1", None),
    "puter": ("gpt-4o-mini", "https://api.puter.com/puterai/openai/v1", None),
    "sambanova": ("DeepSeek-V3.1", "https://api.sambanova.ai/v1", None),
    "scaleway": ("qwen3-235b-a22b-instruct-2507", "https://api.scaleway.ai/v1", None),
    "upstage": ("solar-pro3", "https://api.upstage.ai/v1", None),
    "v0-vercel": ("v0-1.5-md", "https://api.v0.dev/v1", None),
    "venice": ("venice-latest", "https://api.venice.ai/api/v1", None),
    "vercel-ai-gateway": ("openai/gpt-4.1", "https://ai-gateway.vercel.sh/v1", None),
    "volcengine": ("deepseek-v3-2-251201", "https://ark.cn-beijing.volces.com/api/v3", None),
    "wandb": ("openai/gpt-oss-120b", "https://api.inference.wandb.ai/v1", None),
    "xiaomi-mimo": ("mimo-v2.5-pro", "https://api.xiaomimimo.com/v1", None),
    "ai21": ("jamba-large-1.7", "https://api.ai21.com/studio/v1", None),
}

_ANTHROPIC_BETA_API_KEY = (
    "claude-code-20250219,interleaved-thinking-2025-05-14,"
    "context-management-2025-06-27,prompt-caching-scope-2026-01-05,"
    "advanced-tool-use-2025-11-20,effort-2025-11-24,structured-outputs-2025-12-15,"
    "fast-mode-2026-02-01,redact-thinking-2026-02-12,token-efficient-tools-2026-03-28"
)

# Anthropic Messages API shape (MiniMax / Z.ai GLM / Kimi Coding / Bailian coding plan).
_ANTHROPIC_MESSAGES_PROBES: dict[str, tuple[str, str, bool]] = {
    "glm": ("https://api.z.ai/api/anthropic/v1/messages?beta=true", "glm-5.1", False),
    "glmt": ("https://api.z.ai/api/anthropic/v1/messages?beta=true", "glm-5.1", False),
    "zai": ("https://api.z.ai/api/anthropic/v1/messages?beta=true", "glm-5.1", False),
    "kimi-coding-apikey": ("https://api.kimi.com/coding/v1/messages?beta=true", "kimi-k2.5", False),
    "bailian-coding-plan": (
        "https://coding-intl.dashscope.aliyuncs.com/apps/anthropic/v1/messages?beta=true",
        "qwen3.5-plus",
        False,
    ),
    "minimax": ("https://api.minimax.io/anthropic/v1/messages?beta=true", "MiniMax-M2.7", True),
    "minimax-cn": ("https://api.minimaxi.com/anthropic/v1/messages?beta=true", "MiniMax-M2.7", True),
}

_ENV_ONLY_PROVIDERS = frozenset({"bytez", "cloudflare-ai", "gigachat", "nlpcloud", "petals"})


def _mask_key(key: str) -> str:
    if len(key) <= 20:
        return key
    return f"{key[:10]}...{key[-10:]}"


def _extract_error_code(body: object, fallback: str) -> str:
    if isinstance(body, dict):
        code = body.get("code")
        if code:
            return str(code)
        err = body.get("error")
        if isinstance(err, dict) and err.get("code"):
            return str(err["code"])
    return fallback


def _log_success(provider: str, key: str, detail: str = "ok") -> str:
    rich.print(f"🔑 [bold green]available {provider} key[/bold green]: [orange_red1]'{_mask_key(key)}'[/orange_red1] ({detail})")
    return "yes"


def _openai_compatible_check(
    key: str,
    provider: str,
    model: str,
    base_url: str | None = None,
    default_headers: dict[str, str] | None = None,
) -> str:
    try:
        client_kwargs: dict[str, object] = {"api_key": key, "base_url": base_url}
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        client = OpenAI(**client_kwargs)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Reply with lowercase yes only."},
                {"role": "user", "content": "yes or no? say yes"},
            ],
            max_tokens=1,
        )
        result = completion.choices[0].message.content or "yes"
        return _log_success(provider, key, result)
    except AuthenticationError as exc:
        code = _extract_error_code(getattr(exc, "body", None), "invalid_api_key")
        rich.print(f"[deep_sky_blue1]{provider}:{code} ({exc.status_code})[/deep_sky_blue1]: '{_mask_key(key)}'")  # type: ignore[arg-type]
        return code
    except RateLimitError as exc:
        code = _extract_error_code(getattr(exc, "body", None), "rate_limit")
        rich.print(f"[deep_sky_blue1]{provider}:{code} ({exc.status_code})[/deep_sky_blue1]: '{_mask_key(key)}'")  # type: ignore[arg-type]
        return code
    except APIStatusError as exc:
        code = _extract_error_code(getattr(exc, "body", None), f"http_{exc.status_code}")
        rich.print(f"[bold red]{provider}:{code} ({exc.status_code})[/bold red]: '{_mask_key(key)}'")
        return code
    except Exception as exc:  # pylint: disable=broad-except
        rich.print(f"[bold red]{provider}:{exc}[/bold red]: '{_mask_key(key)}'")
        return "Unknown Error"


def _google_gemini_check(key: str) -> str:
    """
    Google AI Studio validation:
    GET /v1beta/models with x-goog-api-key.
    """
    req = request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models",
        headers={"x-goog-api-key": key},
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=DEFAULT_TIMEOUT):
            return _log_success("google", key)
    except error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="ignore")
        except Exception:  # pylint: disable=broad-except
            pass
        normalized_body = body.lower()
        if exc.code in {401, 403} or "api_key_invalid" in normalized_body or "api key not valid" in normalized_body:
            rich.print(f"[deep_sky_blue1]google:invalid_api_key ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "invalid_api_key"
        if exc.code == 429:
            rich.print(f"[deep_sky_blue1]google:rate_limit ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "rate_limit"
        rich.print(f"[bold red]google:http_{exc.code}[/bold red]: '{_mask_key(key)}'")
        return f"http_{exc.code}"
    except Exception as exc:  # pylint: disable=broad-except
        rich.print(f"[bold red]google:{exc}[/bold red]: '{_mask_key(key)}'")
        return "Unknown Error"


def _anthropic_check(key: str, model: str = "claude-3-5-haiku-latest") -> str:
    payload = json.dumps(
        {
            "model": model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "test"}],
        }
    ).encode("utf-8")
    req = request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=DEFAULT_TIMEOUT):
            return _log_success("anthropic", key)
    except error.HTTPError as exc:
        if exc.code in {401, 403}:
            rich.print(f"[deep_sky_blue1]anthropic:invalid_api_key ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "invalid_api_key"
        if exc.code == 429:
            rich.print(f"[deep_sky_blue1]anthropic:rate_limit ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "rate_limit"
        # 4xx non-auth often means request/model issue but auth passed.
        if 400 <= exc.code < 500:
            return _log_success("anthropic", key, f"http_{exc.code}")
        rich.print(f"[bold red]anthropic:http_{exc.code}[/bold red]: '{_mask_key(key)}'")
        return f"http_{exc.code}"
    except Exception as exc:  # pylint: disable=broad-except
        rich.print(f"[bold red]anthropic:{exc}[/bold red]: '{_mask_key(key)}'")
        return "Unknown Error"


def _perplexity_search_check(key: str) -> str:
    payload = json.dumps({"query": "test", "max_results": 1}).encode("utf-8")
    req = request.Request(
        "https://api.perplexity.ai/search",
        data=payload,
        headers={"content-type": "application/json", "authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=DEFAULT_TIMEOUT):
            return _log_success("perplexity", key)
    except error.HTTPError as exc:
        if exc.code in {401, 403}:
            rich.print(f"[deep_sky_blue1]perplexity:invalid_api_key ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "invalid_api_key"
        if exc.code == 429:
            rich.print(f"[deep_sky_blue1]perplexity:rate_limit ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "rate_limit"
        if 400 <= exc.code < 500:
            return _log_success("perplexity", key, f"http_{exc.code}")
        rich.print(f"[bold red]perplexity:http_{exc.code}[/bold red]: '{_mask_key(key)}'")
        return f"http_{exc.code}"
    except Exception as exc:  # pylint: disable=broad-except
        rich.print(f"[bold red]perplexity:{exc}[/bold red]: '{_mask_key(key)}'")
        return "Unknown Error"


def _tavily_search_check(key: str) -> str:
    payload = json.dumps({"query": "test", "max_results": 1}).encode("utf-8")
    req = request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"content-type": "application/json", "authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=DEFAULT_TIMEOUT):
            return _log_success("tavily", key)
    except error.HTTPError as exc:
        if exc.code in {401, 403}:
            rich.print(f"[deep_sky_blue1]tavily:invalid_api_key ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "invalid_api_key"
        if exc.code == 429:
            rich.print(f"[deep_sky_blue1]tavily:rate_limit ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "rate_limit"
        if 400 <= exc.code < 500:
            return _log_success("tavily", key, f"http_{exc.code}")
        rich.print(f"[bold red]tavily:http_{exc.code}[/bold red]: '{_mask_key(key)}'")
        return f"http_{exc.code}"
    except Exception as exc:  # pylint: disable=broad-except
        rich.print(f"[bold red]tavily:{exc}[/bold red]: '{_mask_key(key)}'")
        return "Unknown Error"


def _huggingface_check(key: str) -> str:
    req = request.Request(
        "https://huggingface.co/api/whoami-v2",
        headers={"authorization": f"Bearer {key}"},
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=DEFAULT_TIMEOUT):
            return _log_success("huggingface", key)
    except error.HTTPError as exc:
        if exc.code in {401, 403}:
            rich.print(f"[deep_sky_blue1]huggingface:invalid_api_key ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "invalid_api_key"
        if exc.code == 429:
            rich.print(f"[deep_sky_blue1]huggingface:rate_limit ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "rate_limit"
        rich.print(f"[bold red]huggingface:http_{exc.code}[/bold red]: '{_mask_key(key)}'")
        return f"http_{exc.code}"
    except Exception as exc:  # pylint: disable=broad-except
        rich.print(f"[bold red]huggingface:{exc}[/bold red]: '{_mask_key(key)}'")
        return "Unknown Error"


def _deepgram_check(key: str) -> str:
    req = request.Request(
        "https://api.deepgram.com/v1/auth/token",
        headers={"authorization": f"Token {key}"},
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=DEFAULT_TIMEOUT):
            return _log_success("deepgram", key)
    except error.HTTPError as exc:
        if exc.code in {401, 403}:
            rich.print(f"[deep_sky_blue1]deepgram:invalid_api_key ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "invalid_api_key"
        if exc.code == 429:
            rich.print(f"[deep_sky_blue1]deepgram:rate_limit ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "rate_limit"
        if 400 <= exc.code < 500:
            return _log_success("deepgram", key, f"http_{exc.code}")
        rich.print(f"[bold red]deepgram:http_{exc.code}[/bold red]: '{_mask_key(key)}'")
        return f"http_{exc.code}"
    except Exception as exc:  # pylint: disable=broad-except
        rich.print(f"[bold red]deepgram:{exc}[/bold red]: '{_mask_key(key)}'")
        return "Unknown Error"


def _serper_search_check(key: str) -> str:
    payload = json.dumps({"q": "test", "num": 1}).encode("utf-8")
    req = request.Request(
        "https://google.serper.dev/search",
        data=payload,
        headers={"content-type": "application/json", "x-api-key": key},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=DEFAULT_TIMEOUT):
            return _log_success("serper-search", key)
    except error.HTTPError as exc:
        if exc.code in {401, 403}:
            rich.print(f"[deep_sky_blue1]serper-search:invalid_api_key ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "invalid_api_key"
        if exc.code == 429:
            rich.print(f"[deep_sky_blue1]serper-search:rate_limit ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "rate_limit"
        if 400 <= exc.code < 500:
            return _log_success("serper-search", key, f"http_{exc.code}")
        rich.print(f"[bold red]serper-search:http_{exc.code}[/bold red]: '{_mask_key(key)}'")
        return f"http_{exc.code}"
    except Exception as exc:  # pylint: disable=broad-except
        rich.print(f"[bold red]serper-search:{exc}[/bold red]: '{_mask_key(key)}'")
        return "Unknown Error"


def _exa_search_check(key: str) -> str:
    payload = json.dumps({"query": "test", "numResults": 1}).encode("utf-8")
    req = request.Request(
        "https://api.exa.ai/search",
        data=payload,
        headers={"content-type": "application/json", "x-api-key": key},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=DEFAULT_TIMEOUT):
            return _log_success("exa-search", key)
    except error.HTTPError as exc:
        if exc.code in {401, 403}:
            rich.print(f"[deep_sky_blue1]exa-search:invalid_api_key ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "invalid_api_key"
        if exc.code == 429:
            rich.print(f"[deep_sky_blue1]exa-search:rate_limit ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "rate_limit"
        if 400 <= exc.code < 500:
            return _log_success("exa-search", key, f"http_{exc.code}")
        rich.print(f"[bold red]exa-search:http_{exc.code}[/bold red]: '{_mask_key(key)}'")
        return f"http_{exc.code}"
    except Exception as exc:  # pylint: disable=broad-except
        rich.print(f"[bold red]exa-search:{exc}[/bold red]: '{_mask_key(key)}'")
        return "Unknown Error"


def _brave_search_check(key: str) -> str:
    req = request.Request(
        "https://api.search.brave.com/res/v1/web/search?q=test&count=1",
        headers={"accept": "application/json", "x-subscription-token": key},
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=DEFAULT_TIMEOUT):
            return _log_success("brave-search", key)
    except error.HTTPError as exc:
        if exc.code in {401, 403}:
            rich.print(f"[deep_sky_blue1]brave-search:invalid_api_key ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "invalid_api_key"
        if exc.code == 429:
            rich.print(f"[deep_sky_blue1]brave-search:rate_limit ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "rate_limit"
        if 400 <= exc.code < 500:
            return _log_success("brave-search", key, f"http_{exc.code}")
        rich.print(f"[bold red]brave-search:http_{exc.code}[/bold red]: '{_mask_key(key)}'")
        return f"http_{exc.code}"
    except Exception as exc:  # pylint: disable=broad-except
        rich.print(f"[bold red]brave-search:{exc}[/bold red]: '{_mask_key(key)}'")
        return "Unknown Error"


def _assemblyai_check(key: str) -> str:
    req = request.Request(
        "https://api.assemblyai.com/v2/transcript?limit=1",
        headers={"authorization": key, "content-type": "application/json"},
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=DEFAULT_TIMEOUT):
            return _log_success("assemblyai", key)
    except error.HTTPError as exc:
        if exc.code in {401, 403}:
            rich.print(f"[deep_sky_blue1]assemblyai:invalid_api_key ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "invalid_api_key"
        if exc.code == 429:
            rich.print(f"[deep_sky_blue1]assemblyai:rate_limit ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "rate_limit"
        if 400 <= exc.code < 500:
            return _log_success("assemblyai", key, f"http_{exc.code}")
        rich.print(f"[bold red]assemblyai:http_{exc.code}[/bold red]: '{_mask_key(key)}'")
        return f"http_{exc.code}"
    except Exception as exc:  # pylint: disable=broad-except
        rich.print(f"[bold red]assemblyai:{exc}[/bold red]: '{_mask_key(key)}'")
        return "Unknown Error"


def _elevenlabs_check(key: str) -> str:
    req = request.Request(
        "https://api.elevenlabs.io/v1/voices",
        headers={"xi-api-key": key, "content-type": "application/json"},
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=DEFAULT_TIMEOUT):
            return _log_success("elevenlabs", key)
    except error.HTTPError as exc:
        if exc.code in {401, 403}:
            rich.print(f"[deep_sky_blue1]elevenlabs:invalid_api_key ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "invalid_api_key"
        if exc.code == 429:
            rich.print(f"[deep_sky_blue1]elevenlabs:rate_limit ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "rate_limit"
        if 400 <= exc.code < 500:
            return _log_success("elevenlabs", key, f"http_{exc.code}")
        rich.print(f"[bold red]elevenlabs:http_{exc.code}[/bold red]: '{_mask_key(key)}'")
        return f"http_{exc.code}"
    except Exception as exc:  # pylint: disable=broad-except
        rich.print(f"[bold red]elevenlabs:{exc}[/bold red]: '{_mask_key(key)}'")
        return "Unknown Error"


def _linkup_search_check(key: str) -> str:
    payload = json.dumps(
        {"q": "test", "depth": "standard", "outputType": "searchResults", "maxResults": 1}
    ).encode("utf-8")
    req = request.Request(
        "https://api.linkup.so/v1/search",
        data=payload,
        headers={"content-type": "application/json", "authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=DEFAULT_TIMEOUT):
            return _log_success("linkup-search", key)
    except error.HTTPError as exc:
        if exc.code in {401, 403}:
            rich.print(f"[deep_sky_blue1]linkup-search:invalid_api_key ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "invalid_api_key"
        if exc.code == 429:
            rich.print(f"[deep_sky_blue1]linkup-search:rate_limit ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "rate_limit"
        if 400 <= exc.code < 500:
            return _log_success("linkup-search", key, f"http_{exc.code}")
        rich.print(f"[bold red]linkup-search:http_{exc.code}[/bold red]: '{_mask_key(key)}'")
        return f"http_{exc.code}"
    except Exception as exc:  # pylint: disable=broad-except
        rich.print(f"[bold red]linkup-search:{exc}[/bold red]: '{_mask_key(key)}'")
        return "Unknown Error"


def _searchapi_search_check(key: str) -> str:
    url = f"https://www.searchapi.io/api/v1/search?engine=google&q=test&api_key={quote(key, safe='')}"
    req = request.Request(url, headers={"accept": "application/json"}, method="GET")
    try:
        with request.urlopen(req, timeout=DEFAULT_TIMEOUT):
            return _log_success("searchapi-search", key)
    except error.HTTPError as exc:
        if exc.code in {401, 403}:
            rich.print(
                f"[deep_sky_blue1]searchapi-search:invalid_api_key ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'"
            )
            return "invalid_api_key"
        if exc.code == 429:
            rich.print(f"[deep_sky_blue1]searchapi-search:rate_limit ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "rate_limit"
        if 400 <= exc.code < 500:
            return _log_success("searchapi-search", key, f"http_{exc.code}")
        rich.print(f"[bold red]searchapi-search:http_{exc.code}[/bold red]: '{_mask_key(key)}'")
        return f"http_{exc.code}"
    except Exception as exc:  # pylint: disable=broad-except
        rich.print(f"[bold red]searchapi-search:{exc}[/bold red]: '{_mask_key(key)}'")
        return "Unknown Error"


def _youcom_search_check(key: str) -> str:
    req = request.Request(
        "https://ydc-index.io/v1/search?query=test&count=1",
        headers={"accept": "application/json", "x-api-key": key},
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=DEFAULT_TIMEOUT):
            return _log_success("youcom-search", key)
    except error.HTTPError as exc:
        if exc.code in {401, 403}:
            rich.print(f"[deep_sky_blue1]youcom-search:invalid_api_key ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "invalid_api_key"
        if exc.code == 429:
            rich.print(f"[deep_sky_blue1]youcom-search:rate_limit ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "rate_limit"
        if 400 <= exc.code < 500:
            return _log_success("youcom-search", key, f"http_{exc.code}")
        rich.print(f"[bold red]youcom-search:http_{exc.code}[/bold red]: '{_mask_key(key)}'")
        return f"http_{exc.code}"
    except Exception as exc:  # pylint: disable=broad-except
        rich.print(f"[bold red]youcom-search:{exc}[/bold red]: '{_mask_key(key)}'")
        return "Unknown Error"


def _cohere_check(key: str) -> str:
    payload = json.dumps(
        {
            "model": "command-r-08-2024",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1,
        }
    ).encode("utf-8")
    req = request.Request(
        "https://api.cohere.com/v2/chat",
        data=payload,
        headers={"content-type": "application/json", "authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=DEFAULT_TIMEOUT):
            return _log_success("cohere", key)
    except error.HTTPError as exc:
        if exc.code in {401, 403}:
            rich.print(f"[deep_sky_blue1]cohere:invalid_api_key ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "invalid_api_key"
        if exc.code == 429:
            rich.print(f"[deep_sky_blue1]cohere:rate_limit ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "rate_limit"
        if 400 <= exc.code < 500:
            return _log_success("cohere", key, f"http_{exc.code}")
        rich.print(f"[bold red]cohere:http_{exc.code}[/bold red]: '{_mask_key(key)}'")
        return f"http_{exc.code}"
    except Exception as exc:  # pylint: disable=broad-except
        rich.print(f"[bold red]cohere:{exc}[/bold red]: '{_mask_key(key)}'")
        return "Unknown Error"


def _anthropic_messages_api_check(key: str, provider: str, url: str, model: str, bearer_auth: bool) -> str:
    payload = json.dumps(
        {"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": "test"}]}
    ).encode("utf-8")
    headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": _ANTHROPIC_BETA_API_KEY,
    }
    if bearer_auth:
        headers["authorization"] = f"Bearer {key}"
    else:
        headers["x-api-key"] = key
    req = request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=DEFAULT_TIMEOUT):
            return _log_success(provider, key)
    except error.HTTPError as exc:
        if exc.code in {401, 403}:
            rich.print(f"[deep_sky_blue1]{provider}:invalid_api_key ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "invalid_api_key"
        if exc.code == 429:
            rich.print(f"[deep_sky_blue1]{provider}:rate_limit ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "rate_limit"
        if 400 <= exc.code < 500:
            return _log_success(provider, key, f"http_{exc.code}")
        rich.print(f"[bold red]{provider}:http_{exc.code}[/bold red]: '{_mask_key(key)}'")
        return f"http_{exc.code}"
    except Exception as exc:  # pylint: disable=broad-except
        rich.print(f"[bold red]{provider}:{exc}[/bold red]: '{_mask_key(key)}'")
        return "Unknown Error"


def _maritalk_check(key: str) -> str:
    payload = json.dumps(
        {"model": "sabia-4", "max_tokens": 1, "messages": [{"role": "user", "content": "test"}]}
    ).encode("utf-8")
    req = request.Request(
        "https://chat.maritaca.ai/api/chat/completions",
        data=payload,
        headers={"authorization": f"Key {key}", "content-type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=DEFAULT_TIMEOUT):
            return _log_success("maritalk", key)
    except error.HTTPError as exc:
        if exc.code in {401, 403}:
            rich.print(f"[deep_sky_blue1]maritalk:invalid_api_key ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "invalid_api_key"
        if exc.code == 429:
            rich.print(f"[deep_sky_blue1]maritalk:rate_limit ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "rate_limit"
        if 400 <= exc.code < 500:
            return _log_success("maritalk", key, f"http_{exc.code}")
        rich.print(f"[bold red]maritalk:http_{exc.code}[/bold red]: '{_mask_key(key)}'")
        return f"http_{exc.code}"
    except Exception as exc:  # pylint: disable=broad-except
        rich.print(f"[bold red]maritalk:{exc}[/bold red]: '{_mask_key(key)}'")
        return "Unknown Error"


def _gitlab_pat_check(key: str) -> str:
    req = request.Request(
        "https://gitlab.com/api/v4/code_suggestions/direct_access",
        data=b"{}",
        headers={"content-type": "application/json", "authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=DEFAULT_TIMEOUT):
            return _log_success("gitlab", key)
    except error.HTTPError as exc:
        if exc.code == 401:
            rich.print(f"[deep_sky_blue1]gitlab:invalid_api_key ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "invalid_api_key"
        if exc.code == 429:
            rich.print(f"[deep_sky_blue1]gitlab:rate_limit ({exc.code})[/deep_sky_blue1]: '{_mask_key(key)}'")
            return "rate_limit"
        if 400 <= exc.code < 500:
            return _log_success("gitlab", key, f"http_{exc.code}")
        rich.print(f"[bold red]gitlab:http_{exc.code}[/bold red]: '{_mask_key(key)}'")
        return f"http_{exc.code}"
    except Exception as exc:  # pylint: disable=broad-except
        rich.print(f"[bold red]gitlab:{exc}[/bold red]: '{_mask_key(key)}'")
        return "Unknown Error"


def _env_only_no_probe(provider: str, key: str) -> str:
    rich.print(f"[dark_orange]{provider}:no_probe (env discovery only)[/dark_orange]: '{_mask_key(key)}'")
    return "no_probe"


def detect_provider(key: str) -> str:
    for pattern in PROVIDER_PATTERNS:
        if pattern["regex"].fullmatch(key):
            return pattern["provider"]
    if key.startswith(("sk-proj-", "sk-svcacct-")):
        return "openai"
    return "unknown"


def check_key(key: str, model: str = "gpt-4o-mini", provider: str | None = None) -> str:
    """
    Check key validity with provider-specific logic.
    """
    resolved_provider = provider or detect_provider(key)

    if resolved_provider in _ENV_ONLY_PROVIDERS:
        return _env_only_no_probe(resolved_provider, key)

    if resolved_provider == "reka":
        return _openai_compatible_check(
            key,
            "reka",
            "reka-flash",
            base_url="https://api.reka.ai/v1",
            default_headers={"X-Api-Key": key},
        )

    amp = _ANTHROPIC_MESSAGES_PROBES.get(resolved_provider)
    if amp:
        url_amp, model_amp, bearer_amp = amp
        return _anthropic_messages_api_check(key, resolved_provider, url_amp, model_amp, bearer_amp)

    if resolved_provider == "gitlab":
        return _gitlab_pat_check(key)
    if resolved_provider == "maritalk":
        return _maritalk_check(key)

    if resolved_provider == "google":
        return _google_gemini_check(key)
    if resolved_provider == "anthropic":
        return _anthropic_check(key)
    if resolved_provider == "cohere":
        return _cohere_check(key)
    if resolved_provider in {"perplexity", "perplexity-search"}:
        return _perplexity_search_check(key)
    if resolved_provider in {"tavily", "tavily-search"}:
        return _tavily_search_check(key)
    if resolved_provider == "huggingface":
        return _huggingface_check(key)
    if resolved_provider == "deepgram":
        return _deepgram_check(key)
    if resolved_provider == "assemblyai":
        return _assemblyai_check(key)
    if resolved_provider == "elevenlabs":
        return _elevenlabs_check(key)
    if resolved_provider == "serper-search":
        return _serper_search_check(key)
    if resolved_provider == "exa-search":
        return _exa_search_check(key)
    if resolved_provider == "brave-search":
        return _brave_search_check(key)
    if resolved_provider == "linkup-search":
        return _linkup_search_check(key)
    if resolved_provider == "searchapi-search":
        return _searchapi_search_check(key)
    if resolved_provider == "youcom-search":
        return _youcom_search_check(key)

    probe = _OPENAI_COMPAT_PROBES.get(resolved_provider)
    if probe:
        model_id, base_url, hdrs = probe
        return _openai_compatible_check(
            key,
            resolved_provider,
            model_id,
            base_url=base_url,
            default_headers=hdrs,
        )

    # default behavior: OpenAI validation (keeps backward compatibility)
    return _openai_compatible_check(key, provider="openai", model=model)


if __name__ == "__main__":
    check_key("sk-proj-12345")
