from gretel_client.gretel.config_setup import smart_load_yaml

from navigator_helpers.pipelines.config.base import ConfigLike, PipelineConfig
from navigator_helpers.pipelines.config.text_to_code import (
    NL2CodeAutoConfig,
    NL2CodeManualConfig,
)


def smart_load_pipeline_config(config: ConfigLike) -> PipelineConfig:
    if not isinstance(config, (NL2CodeManualConfig, NL2CodeAutoConfig)):
        config = smart_load_yaml(config)
        config = (
            NL2CodeAutoConfig(**config)
            if "num_domains" in config
            else NL2CodeManualConfig(**config)
        )
    return config
