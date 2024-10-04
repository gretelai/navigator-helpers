import json
import logging

from typing import Optional, Tuple, TYPE_CHECKING

from gretel_client import Gretel
from gretel_client.inference_api.base import InferenceAPIModelType

from navigator_helpers.data_models import ContextualTag, ContextualTags, DataModel
from navigator_helpers.text_inference import TextInference

from .prompts import (
    DATA_MODEL_GENERATION_PROMPT,
    EXAMPLE_YAML,
    TAG_EXTRACTION_INITIAL_USER_PROMPT,
    TAG_EXTRACTION_ITERATION_USER_PROMPT,
    TAG_EXTRACTION_SYSTEM_PROMPT,
)
from .utils import (
    calculate_diversity,
    extract_json_from_response,
    parse_yaml,
    safe_variable_name,
)

if TYPE_CHECKING:
    from gretel_client.inference_api.natural_language import NaturalLanguageInferenceAPI

logger = logging.getLogger(__name__)


class ConfigGenerator:
    def __init__(
        self, api_key: Optional[str] = "prompt", model: str = "gretelai/gpt-auto"
    ):
        self.gretel = Gretel(api_key=api_key, validate=True)
        self.llm: NaturalLanguageInferenceAPI = (
            self.gretel.factories.initialize_navigator_api(
                InferenceAPIModelType.NATURAL_LANGUAGE, backend_model=model
            )
        )  # type: ignore
        self.text_inference = TextInference(self.llm)
        self.user_task: Optional[str] = None
        self.tags: Optional[ContextualTags] = None
        self.data_model: Optional[DataModel] = None

    def set_user_task(self, user_task: str):
        """Set the user's task description."""
        self.user_task = user_task
        logger.info(f"User task set: {user_task}")

    def generate_tags(self, diversity_target: int = 10000) -> ContextualTags:
        """Generate contextual tags based on the user task."""
        if not self.user_task:
            raise ValueError("User task must be set before generating tags.")
        self.tags = generate_contextual_tags(
            self.user_task, self.text_inference, diversity_target
        )
        return self.tags

    def generate_data_model(self) -> Optional[DataModel]:
        """Generate the data model based on the user task and tags."""
        if not self.user_task or not self.tags:
            raise ValueError(
                "User task and tags must be set before generating the data model."
            )
        self.data_model = generate_data_model(
            self.user_task, EXAMPLE_YAML, self.tags, self.text_inference
        )
        return self.data_model

    def get_config(self) -> str:
        """Get the final configuration as YAML."""
        if not self.data_model:
            raise ValueError("Data model must be generated before getting the config.")
        return self.data_model.to_yaml()


def extract_tags_with_llm(
    description: str, text_inference: TextInference, diversity_target: int = 10000
) -> ContextualTags:
    current_diversity = 0
    iteration = 0

    contextual_tags = ContextualTags()

    while current_diversity < diversity_target:
        iteration += 1
        logger.info(f"Iteration {iteration}: Current diversity = {current_diversity}")

        if iteration == 1:
            user_prompt = TAG_EXTRACTION_INITIAL_USER_PROMPT.format(
                description=description
            )
        else:
            tags_json = json.dumps(
                [tag.model_dump() for tag in contextual_tags.tags], indent=2
            )
            user_prompt = TAG_EXTRACTION_ITERATION_USER_PROMPT.format(
                description=description, tags_json=tags_json
            )

        assistant_reply = text_inference.generate(
            user_prompt,
            use_reflection=False,
            temperature=0.7,
            max_tokens=2000,
            system_message=TAG_EXTRACTION_SYSTEM_PROMPT,
        )

        json_content = extract_json_from_response(assistant_reply)

        try:
            new_tags_data = json.loads(json_content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error: {e}")
            logger.error(f"Assistant reply:\n{assistant_reply}")
            break

        for tag_data in new_tags_data:
            tag_name = safe_variable_name(tag_data["name"])
            new_values = tag_data["values"]

            if existing_tag := contextual_tags.get_tag_by_name(tag_name):
                existing_tag.add_values(new_values)
            else:
                contextual_tags.add_tag(ContextualTag(name=tag_name, values=new_values))

        current_diversity = calculate_diversity(contextual_tags.tags)
        logger.info(f"Current diversity: {current_diversity}")

        if current_diversity >= diversity_target:
            logger.info(f"Diversity target met: {current_diversity}")
            break

    return contextual_tags


def generate_contextual_tags(
    user_task: str, text_inference: TextInference, diversity_target: int = 10000
) -> ContextualTags:
    return extract_tags_with_llm(user_task, text_inference, diversity_target)


def generate_data_model(
    user_task: str,
    example_yaml: str,
    tags: ContextualTags,
    text_inference: TextInference,
    max_retries: int = 3,
) -> Optional[DataModel]:
    prompt = DATA_MODEL_GENERATION_PROMPT.format(
        user_task=user_task, tags_yaml=tags.to_yaml(), example_yaml=example_yaml
    )
    first_llm_response = None
    data_model = None
    attempt = 0

    while attempt < max_retries:
        if attempt > 0:
            logger.info(f"Attempting to repair the YAML (retry {attempt})")

        llm_response = text_inference.generate(
            prompt,
            use_reflection=True,
            temperature=0.7,
            max_tokens=2048,
            field_name="dataset_config",
        )

        if attempt == 0:
            first_llm_response = llm_response

        data_model = parse_yaml(llm_response)

        if data_model:
            for field in data_model.fields:
                field.name = safe_variable_name(field.name)

            data_model.contextual_tags = tags

            logger.info("\nGenerated DataModel:")
            logger.info(data_model.to_yaml())
            return data_model

        attempt += 1

    logger.error("Failed to generate a valid DataModel after maximum retries.")
    if first_llm_response:
        logger.error("Final raw LLM response for debugging:")
        logger.error(first_llm_response)

    return None


def generate_config(
    user_task: str, text_inference: TextInference, max_retries: int = 3
) -> Tuple[Optional[DataModel], ContextualTags]:
    tags = generate_contextual_tags(user_task, text_inference, diversity_target=10000)
    logger.info("\nGenerated Tags and Attributes:")
    logger.info(tags.to_yaml())

    data_model = generate_data_model(
        user_task, EXAMPLE_YAML, tags, text_inference, max_retries
    )

    return data_model, tags
