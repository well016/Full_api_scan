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

# regex, have_many_results (if true, using AND to filter out keywords), result_too_lang (if true, the result needs to be expanded)
REGEX_LIST = [
    # Named Project API Key (no matter normal or restricted) still valid until Dec 2, 2024
    (re.compile(r"sk-proj-[A-Za-z0-9-_]{74}T3BlbkFJ[A-Za-z0-9-_]{73}A"), True, True),

    # Service Account key (new format, valid until Oct 13, 2025)
    (re.compile(r"sk-svcacct-[A-Za-z0-9-_]{74}T3BlbkFJ[A-Za-z0-9-_]{73}A"), False, True),

    # Old Project API Key (as of Oct 13, 2025, not sure if still valid)
    (re.compile(r"sk-proj-[A-Za-z0-9-_]{58}T3BlbkFJ[A-Za-z0-9-_]{58}"), True, True),

    # Service Account Key (not valid since Oct 13, 2025)
    # (re.compile(r"sk-svcacct-[A-Za-z0-9-_]\+T3BlbkFJ[A-Za-z0-9-_]+"), False, False), # No search results on Oct 14, 2025

    (re.compile(r"sk-proj-[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}"), True, False),
    # (re.compile(r"sk-[a-zA-Z0-9]{48}"), True, False), # OpenAI deprecated these format, because the keys are only "sk-proj" "sk-svcacct-"
]
