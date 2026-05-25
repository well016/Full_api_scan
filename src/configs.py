"""
This module is used to store the configurations.
"""

import re

# Keywords are not enabled by current version.
KEYWORDS = [
    "CoT",
    "DPO",
    "RLHF",
    "agent",
    "ai model",
    "aios",
    "api key",
    "apikey",
    "artificial intelligence",
    "chain of thought",
    "chatbot",
    "chatgpt",
    "competitor analysis",
    "content strategy",
    "conversational AI",
    "data analysis",
    "deep learning",
    "direct preference optimization",
    "experiment",
    "gpt",
    "gpt-3",
    "gpt-4",
    "gpt4",
    "key",
    "keyword clustering",
    "keyword research",
    "lab",
    "language model experimentation",
    "large language model",
    "llama.cpp",
    "llm",
    "long-tail keywords",
    "machine learning",
    "multi-agent",
    "multi-agent systems",
    "natural language processing",
    "openai",
    "personalized AI",
    "project",
    "rag",
    "reinforcement learning from human feedback",
    "retrieval-augmented generation",
    "search intent",
    "semantic search",
    "thoughts",
    "virtual assistant",
    "实验",
    "密钥",
    "测试",
    "语言模型",
]

LANGUAGES = [
    "Dotenv",
    "Text",
    "JavaScript",
    "Python",
    "TypeScript",
    "Dockerfile",
    "Markdown",
    '"Jupyter Notebook"',
    "Shell",
    "Java",
    "Go",
    "C%2B%2B",
    "PHP",
]

PATHS = [
    "path:.xml OR path:.json OR path:.properties OR path:.sql OR path:.txt OR path:.log OR path:.tmp OR path:.backup OR path:.bak OR path:.enc",
    "path:.yml OR path:.yaml OR path:.toml OR path:.ini OR path:.config OR path:.conf OR path:.cfg OR path:.env OR path:.envrc OR path:.prod",
    "path:.secret OR path:.private OR path:*.key",
]

# provider aware regex patterns:
# - provider: key provider identifier used by validation logic
# - regex: compiled pattern used for search extraction
# - have_many_results: if true, add language filter in search query
# - result_too_long: if true, key often appears in collapsed snippets and needs page expansion
PROVIDER_PATTERNS = [
    # OpenAI
    {
        "provider": "openai",
        "regex": re.compile(r"sk-proj-[A-Za-z0-9-_]{74}T3BlbkFJ[A-Za-z0-9-_]{73}A"),
        "have_many_results": True,
        "result_too_long": True,
    },
    {
        "provider": "openai",
        "regex": re.compile(r"sk-svcacct-[A-Za-z0-9-_]{74}T3BlbkFJ[A-Za-z0-9-_]{73}A"),
        "have_many_results": False,
        "result_too_long": True,
    },
    {
        "provider": "openai",
        "regex": re.compile(r"sk-proj-[A-Za-z0-9-_]{58}T3BlbkFJ[A-Za-z0-9-_]{58}"),
        "have_many_results": True,
        "result_too_long": True,
    },
    {
        "provider": "openai",
        "regex": re.compile(r"sk-proj-[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}"),
        "have_many_results": True,
        "result_too_long": False,
    },
    # OpenRouter (officially used as Bearer key on https://openrouter.ai/api/v1)
    {
        "provider": "openrouter",
        "regex": re.compile(r"\bsk-or-v1-[A-Fa-f0-9]{64}\b"),
        "have_many_results": True,
        "result_too_long": False,
    },
    # NVIDIA NIM (OpenAI compatible endpoint: https://integrate.api.nvidia.com/v1)
    {
        "provider": "nvidia",
        "regex": re.compile(r"\bnvapi-[A-Za-z0-9_-]{32,}\b"),
        "have_many_results": True,
        "result_too_long": False,
    },
    # xAI Grok API keys (https://api.x.ai)
    {
        "provider": "xai",
        "regex": re.compile(r"\bxai-[A-Za-z0-9_-]{24,}\b"),
        "have_many_results": True,
        "result_too_long": False,
    },
    # Google AI Studio / Gemini API key
    {
        "provider": "google",
        "regex": re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"),
        "have_many_results": True,
        "result_too_long": False,
    },
    # Anthropic
    {
        "provider": "anthropic",
        "regex": re.compile(r"\bsk-ant-(?:api\d{2}-)?[A-Za-z0-9_-]{20,}\b"),
        "have_many_results": True,
        "result_too_long": False,
    },
    # Groq
    {
        "provider": "groq",
        "regex": re.compile(r"\bgsk_[A-Za-z0-9]{40,}\b"),
        "have_many_results": True,
        "result_too_long": False,
    },
    # Perplexity (used by perplexity and perplexity-search in OmniRoute)
    {
        "provider": "perplexity",
        "regex": re.compile(r"\bpplx-[A-Za-z0-9]{16,}\b"),
        "have_many_results": True,
        "result_too_long": False,
    },
    # Tavily (used by tavily-search in OmniRoute)
    {
        "provider": "tavily",
        "regex": re.compile(r"\btvly-[A-Za-z0-9_-]{16,}\b"),
        "have_many_results": True,
        "result_too_long": False,
    },
    # Hugging Face access token
    {
        "provider": "huggingface",
        "regex": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
        "have_many_results": True,
        "result_too_long": False,
    },
]

# Backward-compatible shape used by existing scanner loops:
# (regex, have_many_results, result_too_long)
REGEX_LIST = [(item["regex"], item["have_many_results"], item["result_too_long"]) for item in PROVIDER_PATTERNS]

# Provider-specific env variable names for smart extraction mode.
# Used when secret format is not unique enough for a strict regex.
ENV_PROVIDER_VARIABLES = {
    "openai": ["OPENAI_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "nvidia": ["NVIDIA_API_KEY", "NIM_API_KEY"],
    "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "perplexity": ["PERPLEXITY_API_KEY", "PPLX_API_KEY"],
    "perplexity-search": ["PERPLEXITY_API_KEY", "PPLX_API_KEY"],
    "tavily": ["TAVILY_API_KEY"],
    "tavily-search": ["TAVILY_API_KEY"],
    "huggingface": ["HUGGINGFACE_API_KEY", "HF_TOKEN", "HF_API_TOKEN"],
    "deepgram": ["DEEPGRAM_API_KEY"],
    "assemblyai": ["ASSEMBLYAI_API_KEY"],
    "elevenlabs": ["ELEVENLABS_API_KEY"],
    "exa-search": ["EXA_API_KEY"],
    "serper-search": ["SERPER_API_KEY"],
    "brave-search": ["BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY"],
    "linkup-search": ["LINKUP_API_KEY"],
    "searchapi-search": ["SEARCHAPI_API_KEY", "SEARCH_API_IO_KEY"],
    "youcom-search": ["YOUCOM_API_KEY", "YDC_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "xai": ["XAI_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "together": ["TOGETHER_API_KEY"],
    "fireworks": ["FIREWORKS_API_KEY"],
    "cerebras": ["CEREBRAS_API_KEY"],
    "cohere": ["COHERE_API_KEY"],
    "nebius": ["NEBIUS_API_KEY"],
    "moonshot": ["MOONSHOT_API_KEY"],
    "agentrouter": ["AGENTROUTER_API_KEY"],
    "siliconflow": ["SILICONFLOW_API_KEY"],
    "qianfan": ["QIANFAN_API_KEY"],
    "ollama-cloud": ["OLLAMA_API_KEY"],
    "blackbox": ["BLACKBOX_API_KEY"],
    "meta-llama": ["LLAMA_API_KEY", "META_LLAMA_API_KEY"],
    "publicai": ["PUBLICAI_API_KEY"],
    "hyperbolic": ["HYPERBOLIC_API_KEY"],
    # OmniRoute-style API key providers (dashboard / env)
    "aimlapi": ["AIMLAPI_API_KEY"],
    "alibaba": ["DASHSCOPE_API_KEY", "ALIBABA_CLOUD_API_KEY"],
    "alicode": ["ALICODE_API_KEY", "DASHSCOPE_API_KEY"],
    "alicode-intl": ["ALICODE_INTL_API_KEY", "DASHSCOPE_API_KEY"],
    "baseten": ["BASETEN_API_KEY"],
    "bailian-coding-plan": ["DASHSCOPE_API_KEY", "ALIBABA_CODING_PLAN_API_KEY"],
    "codestral": ["CODESTRAL_API_KEY", "MISTRAL_CODESTRAL_API_KEY"],
    "crof": ["CROF_API_KEY"],
    "deepinfra": ["DEEPINFRA_API_KEY"],
    "featherless-ai": ["FEATHERLESS_API_KEY"],
    "friendliai": ["FRIENDLI_API_KEY"],
    "galadriel": ["GALADRIEL_API_KEY"],
    "glm": ["Z_AI_API_KEY", "GLM_API_KEY"],
    "glm-cn": ["BIGMODEL_API_KEY", "GLM_CN_API_KEY"],
    "glmt": ["Z_AI_API_KEY", "GLMT_API_KEY"],
    "gitlab": ["GITLAB_TOKEN", "GITLAB_API_TOKEN", "GITLAB_PERSONAL_ACCESS_TOKEN"],
    "gigachat": ["GIGACHAT_AUTHORIZATION_KEY", "GIGACHAT_CREDENTIALS"],
    "heroku": ["HEROKU_INFERENCE_API_KEY"],
    "inference-net": ["INFERENCE_NET_API_KEY"],
    "kimi": ["KIMI_API_KEY"],
    "kimi-coding-apikey": ["KIMI_CODING_API_KEY"],
    "lambda-ai": ["LAMBDA_API_KEY"],
    "llamagate": ["LLAMAGATE_API_KEY"],
    "longcat": ["LONGCAT_API_KEY"],
    "maritalk": ["MARITACA_API_KEY", "MARITALK_API_KEY"],
    "minimax": ["MINIMAX_API_KEY"],
    "minimax-cn": ["MINIMAX_CN_API_KEY"],
    "morph": ["MORPH_API_KEY"],
    "nanogpt": ["NANOGPT_API_KEY"],
    "nlpcloud": ["NLP_CLOUD_API_KEY"],
    "nous-research": ["NOUS_API_KEY"],
    "nscale": ["NSCALE_API_KEY"],
    "opencode-go": ["OPENCODE_API_KEY"],
    "opencode-zen": ["OPENCODE_ZEN_API_KEY"],
    "ovhcloud": ["OVH_AI_API_KEY"],
    "petals": ["PETALS_API_KEY"],
    "pollinations": ["POLLINATIONS_API_KEY"],
    "predibase": ["PREDIBASE_API_KEY"],
    "puter": ["PUTER_AUTH_TOKEN", "PUTER_API_KEY"],
    "reka": ["REKA_API_KEY"],
    "sambanova": ["SAMBANOVA_API_KEY"],
    "scaleway": ["SCALEWAY_API_KEY"],
    "cloudflare-ai": ["CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"],
    "upstage": ["UPSTAGE_API_KEY"],
    "v0-vercel": ["V0_API_KEY"],
    "venice": ["VENICE_API_KEY"],
    "vercel-ai-gateway": ["VERCEL_AI_GATEWAY_API_KEY", "AI_GATEWAY_API_KEY"],
    "volcengine": ["VOLCENGINE_API_KEY", "ARK_API_KEY"],
    "wandb": ["WANDB_API_KEY"],
    "xiaomi-mimo": ["XIAOMI_MIMO_API_KEY"],
    "ai21": ["AI21_API_KEY"],
    "bytez": ["BYTEZ_API_KEY"],
    "zai": ["Z_AI_API_KEY"],
}
