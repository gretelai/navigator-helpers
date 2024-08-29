# LLM configuration

This folder contains configuration files for LLMs to be used with `navigator-helpers`.

## Config format
Configuration is using `litellm`'s format, with an extension that allows tagging configs.
Here's an example:
```yaml
- model_name: gretelai-gpt-llama3-1-8b
  litellm_params:
    model: gretelai/gpt-llama3-1-8b
    api_key: os.environ/GRETEL_API_KEY
    api_base: https://api.gretel.ai
  tags:
  # multiple tags are supported, so they can both: indicate LLM-type (llama3.1 8B) in this case AND logical grouping ("base" for base tasks)
  - gretelai/gpt-llama3-1-8b
  - base
```

### Usage

Load configuration from a file and filter by a tag.

```python
from pathlib import Path
from navigator_helpers.llms import init_llms

registry = init_llms(Path("llm_config_prod.yaml"))
llm_configs = registry.find_by_tags({"base"})
```

Using it directly through the `completion` API.
```python
from navigator_helpers.llms import LLMWrapper, str_to_message

llm = LLMWrapper.from_llm_configs(llm_configs)
llm.completion([str_to_message("What's synthetic data?")], temperature=0.5)
```

Using it with the `autogen` library.

```python
from navigator_helpers.llms.autogen import AutogenAdapter
from autogen import AssistantAgent, UserProxyAgent

adapter = AutogenAdapter(llm_configs)
# Note: with how autogen's support for custom model works, we need to wrap this initialization.
assistant = adapter.initialize_agent(
    AssistantAgent("assistant", llm_config=adapter.create_llm_config())
)
user_proxy = UserProxyAgent("user_proxy")
user_proxy.initiate_chat(assistant, message="What's synthetic data?")
```

## Notes
High level:
- Use `litellm` as the main abstraction over LLMs.
- Create our custom wrapper, so we can swap LiteLLM if needed.
  - Perhaps this will also be useful for our execution, as we may access LLMs differently there?
- For simplicity - initially use `litellm`'s config.
  - If needed, introduce some extensions (like e.g. `tags`, etc.)
- When using with `autogen`, translate the config to `autogen`'s format.
- Application code will only interact with LLMs through `tags`.
  - Each tag may have a single or multiple LLMs configured.
- LiteLLM router will be created after filtering.
- Keys required to access the LLM is using pattern from litellm's config (e.g. `os.environ/AZURE_API_KEY`).
- `llm_config_prod.yaml` is a config using prod Gretel Inference endpoint.