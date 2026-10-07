import os


CHAT_PROVIDERS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "default_model": "openai/gpt-oss-120b", "key_env": ("GROQ_API_KEY", "GROQ_KEY")},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "default_model": "openrouter/auto", "key_env": ("OPENROUTER_API_KEY", "OPENROUTER_KEY")},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "default_model": "gemini-3.8-flash", "key_env": ("GEMINI_API_KEY", "GEMINI_KEY")},
    "huggingface": {"base_url": "https://router.huggingface.co/v1", "default_model": "openai/gpt-oss-120b", "key_env": ("HF_TOKEN", "HUGGINGFACE_TOKEN")},
    "cloudflare": {"base_url": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1", "default_model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast", "key_env": ("CF_API_TOKEN", "CF_TOKEN"), "account_env": ("CF_ACCOUNT_ID", "CF_ACCOUNT")},
    "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1", "default_model": "meta/llama-3.3-70b-instruct", "key_env": ("NVIDIA_API_KEY", "NVIDIA_KEY")},
    "mistral": {"base_url": "https://api.mistral.ai/v1", "default_model": "codestral-latest", "key_env": ("MISTRAL_API_KEY",)},
    "zai": {"base_url": "https://api.z.ai/api/paas/v4", "default_model": "glm-5.3", "key_env": ("ZAI_API_KEY", "Z_AI_API_KEY")},
}

SERVICE_PROVIDERS = {
    "jina": {"capabilities": ["web_search", "web_fetch", "embeddings", "rerank"], "key_env": ("JINA_API_KEY", "JINA_KEY")},
    "elevenlabs": {"capabilities": ["text_to_speech", "speech_to_text", "music_generation"], "key_env": ("ELEVENLABS_API_KEY", "ELEVENLABS_KEY"), "default_tts_model": "eleven_v4_turbo"},
    "udio": {"capabilities": ["music_generation"], "key_env": "UDIO_API_KEY", "supported": False, "note": "No official public developer API was found; do not automate the web session cookie."},
}

# Approximate context windows (tokens). Estimates for the context bar —
# check provider docs for exact figures.
CONTEXT_WINDOWS = {
    "codestral": 256000,
    "mistral": 128000,
    "gpt-oss-120b": 131072,
    "llama-3.3-70b": 128000,
    "glm-5.3": 200000,
    "gemini": 1000000,
    "default": 128000,
}


def ContextWindow(Model):
    """Estimated context window (tokens) for a model string."""
    M = (Model or "").lower()
    for Key, Size in CONTEXT_WINDOWS.items():
        if Key != "default" and Key in M:
            return Size
    return CONTEXT_WINDOWS["default"]


def ProviderNames():
    return sorted(CHAT_PROVIDERS)


def DefaultModel(Provider):
    if Provider not in CHAT_PROVIDERS:
        raise ValueError(f"Unknown chat provider: {Provider}")
    return CHAT_PROVIDERS[Provider]["default_model"]


def ResolveModel(Model):
    Model = (Model or "").strip()
    if ":" in Model:
        Provider, ModelId = Model.split(":", 1)
        Provider, ModelId = Provider.strip().lower(), ModelId.strip()
    else:
        Provider, ModelId = os.getenv("AE_DEFAULT_PROVIDER", "mistral").strip().lower(), Model
    if Provider not in CHAT_PROVIDERS:
        raise ValueError(f"Unknown chat provider '{Provider}'. Available: {', '.join(ProviderNames())}")
    return Provider, ModelId or DefaultModel(Provider)


def ChatEndpoint(Model):
    Provider, ModelId = ResolveModel(Model)
    Config = CHAT_PROVIDERS[Provider]
    ApiKey = next((os.getenv(Name) for Name in Config["key_env"] if os.getenv(Name)), None)
    if not ApiKey:
        raise RuntimeError(f"No API credential configured for {Provider}. Set one of: {', '.join(Config['key_env'])}.")
    BaseUrl = Config["base_url"]
    if Provider == "cloudflare":
        AccountId = next((os.getenv(Name) for Name in Config["account_env"] if os.getenv(Name)), None)
        if not AccountId:
            raise RuntimeError(f"No Cloudflare account ID configured. Set one of: {', '.join(Config['account_env'])}.")
        BaseUrl = BaseUrl.format(account_id=AccountId)
    return {"provider": Provider, "model": ModelId, "api_key": ApiKey, "chat_url": f"{BaseUrl.rstrip('/')}/chat/completions"}


def KeyedProviders():
    """Providers with credentials currently configured (for failover)."""
    Keyed = []
    for Name, Config in CHAT_PROVIDERS.items():
        KeyEnv = Config.get("key_env", ())
        if not any(os.getenv(Key) for Key in KeyEnv):
            continue
        AccountEnv = Config.get("account_env")
        if AccountEnv and not any(os.getenv(Key) for Key in AccountEnv):
            continue
        Keyed.append(Name)
    return sorted(Keyed)


def ChatEndpointWithFallback(Model):
    """Resolve ChatEndpoint, failing over to the next keyed provider when the
    requested one has no credentials. Returns (endpoint, note)."""
    Provider, _ = ResolveModel(Model)
    try:
        return ChatEndpoint(Model), ""
    except RuntimeError:
        pass
    for Fallback in KeyedProviders():
        if Fallback == Provider:
            continue
        try:
            Endpoint = ChatEndpoint(f"{Fallback}:{DefaultModel(Fallback)}")
            return Endpoint, f"(failover: {Provider} has no key, using {Fallback})"
        except RuntimeError:
            continue
    raise RuntimeError(
        f"No API credential configured for {Provider}. Set one of: "
        f"{', '.join(CHAT_PROVIDERS[Provider]['key_env'])}. No fallback provider has keys either."
    )