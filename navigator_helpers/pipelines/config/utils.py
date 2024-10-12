from gretel_client.gretel.config_setup import smart_load_yaml

from navigator_helpers.pipelines.config.base import ConfigLike, PipelineConfig
from navigator_helpers.pipelines.config.text_to_code import (
    NL2CodeAutoConfig,
    NL2CodeManualConfig,
)
from navigator_helpers.pipelines.config.pii_documents import (
    PiiDocsAutoConfig,
    PiiDocsManualConfig,
)


# Mapping of conditions to their corresponding config classes
CONFIG_MAP = {
    ("num_domains", "num_doctypes_per_domain"): PiiDocsAutoConfig,
    ("num_topics_per_domain", "num_complexity_levels"): NL2CodeAutoConfig,
    ("domain_and_doctypes",): PiiDocsManualConfig,
}

def smart_load_pipeline_config(config: ConfigLike) -> PipelineConfig:
    if not isinstance(config, (NL2CodeManualConfig, NL2CodeAutoConfig, PiiDocsManualConfig, PiiDocsAutoConfig)):
        config = smart_load_yaml(config)
        config_class = get_config_class(config)
        return config_class(**config)

    return config

def get_config_class(config: dict) -> type[PipelineConfig]:
    """Determine the appropriate config class based on the keys in the config dictionary."""
    for keys, config_class in CONFIG_MAP.items():
        if all(key in config for key in keys):
            return config_class
    return NL2CodeManualConfig  # Default config class if no match is found
