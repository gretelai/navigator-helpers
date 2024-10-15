from __future__ import annotations

import logging
import random
import re
import uuid

from enum import Enum
from typing import Optional

from tqdm import tqdm

from navigator_helpers.llms.llm_suite import GretelLLMSuite
from navigator_helpers.tasks.base import BaseTaskSuite
from navigator_helpers.tasks.prompt_templates import load_prompt_template_suite
from navigator_helpers.tasks.prompt_templates.template_suite import PromptTemplateSuite
from navigator_helpers.tasks.pii_documents import utils
from navigator_helpers.tasks.pii_documents.pii_generator import PIIGenerator


logger = logging.getLogger(__name__)

PBAR_TEMPLATE = "Running Pipeline [current task: {}]".format

pii_prompts = load_prompt_template_suite("pii_doc")


class DocLang(str, Enum):
    PII_DOC = "pii_doc"

    @property
    def title(self) -> str:
        return self.value.capitalize()


class PiiDocsTaskSuite(BaseTaskSuite):  # Now inherits from BaseTaskSuite
    doc_lang: DocLang
    prompts: PromptTemplateSuite

    def __init__(
        self,
        llm_suite: GretelLLMSuite,
        doc_lang: DocLang = DocLang.PII_DOC,
        **kwargs,
    ):
        # Inherit the initialization from BaseTaskSuite
        super().__init__(llm_suite)

        self.doc_lang = DocLang(doc_lang)
        self.prompts = load_prompt_template_suite(self.doc_lang.value)

        locales = ["en_US", "en_GB", "en_CA", "en_AU", "en_IN"]
        self.pii_generator = PIIGenerator(locales=locales)
        self.allowed_pii_types = self.pii_generator.get_all_pii_generators()
        self.num_pii_types = 4
        self.num_pii_types_repeated = 2

    def generate_domains(self, num_domains: int = 10) -> list[str]:
        response = self.llm_suite.nl_generate(self.prompts.domains(num_domains=num_domains))
        return utils.parse_json_str(response) or []

    def generate_doctypes_from_domains(
        self, domain_list: list[str], num_doctypes_per_domain: int = 5
    ) -> dict[str, list[str]]:
        doctypes = {}
        for domain in domain_list:
            # Log individual domain generation message
            logger.info(f"🔖 Generating document types for domain: {domain}")

            response = self.llm_suite.nl_generate(
                self.prompts.doctypes_from_domains(
                    num_doctypes=num_doctypes_per_domain, domain=domain
                )
            )
            doctypes[domain] = utils.parse_json_str(response) or {}
        return doctypes

    def generate_document_description(
        self, domain: str, doctype: str
    ) -> str:
        response = self.llm_suite.nl_generate(
            self.prompts.document_description_generation(
                domain=domain,
                doctype=doctype,
            )
        )
        return response.strip('"')

    def generate_pii_values(self, pii_types: list) -> list:
        if not pii_types:
            return []

        valid_pii_types = [
            pii_type.strip().lower() for pii_type in pii_types
            if pii_type.strip().lower() in self.allowed_pii_types
        ]

        pii_values = []
        for pii_type in valid_pii_types:
            pii_type_key = re.sub(r'[\s-]', '_', pii_type.lower())
            try:
                pii_value = self.pii_generator.sample(pii_type_key, 1)
                if pii_value and pii_value[0]:
                    pii_values.append({
                        "entity": pii_value[0],
                        "types": [pii_type_key]
                    })
            except Exception as e:
                logger.error(f"Error sampling PII type '{pii_type_key}': {e}")

        return pii_values

    def generate_pii_entities(self, docdesc: str) -> dict:
        response = self.llm_suite.nl_generate(
            self.prompts.pii_type_generation(
                docdesc=docdesc,
                num_pii_types=self.num_pii_types,
                allowed_pii_types=", ".join(self.allowed_pii_types)
            )
        )
        unique_pii_types = utils.parse_json_str(response)
        selected_pii_types = random.sample(unique_pii_types, random.randint(0, len(unique_pii_types)))

        repeated_pii_list = [
            pii_type for pii_type in selected_pii_types
            for _ in range(random.randint(1, self.num_pii_types_repeated))
        ]

        remaining_pii_types = [pii_type for pii_type in unique_pii_types if pii_type not in selected_pii_types]
        pii_types = repeated_pii_list + remaining_pii_types

        return self.generate_pii_values(pii_types)

    def generate_documents(
        self, domain: str, doctype: str, docdesc: str, entities: dict
    ) -> str:
        response = self.llm_suite.nl_generate(
            self.prompts.document_generation(
                domain=domain,
                doctype=doctype,
                docdesc=docdesc,
                entities=entities,
            )
        )
        return response.strip('"')

    def validate_entities(self, text: str, entities: list) -> str:
        missing_entities = []

        # Check if each entity is present in the text
        for entity in entities:
            entity_value = entity['entity']
            entity_types = entity['types']
            if entity_value not in text:
                # Append the entity types to the missing_entities list
                missing_entities.extend(entity_types)

        # If all entities are found, return "valid"
        if not missing_entities:
            return "passed"
        else:
            # Return a message showing the entities that are not found
            missing_entities_str = ", ".join(missing_entities)
            return f"Missing entities: {missing_entities_str}"

    def create_record(
        self,
        domain: str,
        doctype: str,
        entity_validation: bool,
        progress_bar: Optional[tqdm] = None,
    ) -> dict:
        docdesc = self.generate_document_description(domain, doctype)
        pii_entities = self.generate_pii_entities(docdesc)
        doctext = self.generate_documents(
            domain,
            doctype,
            docdesc,
            pii_entities,
        )

        record = {
            "uid": uuid.uuid4().hex,
            "domain": domain,
            "document_type": doctype,
            "document_description": docdesc,
            "entities": pii_entities,
            "text": doctext,
        }

        if entity_validation:
            entity_validation = self.validate_entities(doctext, pii_entities)
            record["entity_validation"] = entity_validation

        return record
