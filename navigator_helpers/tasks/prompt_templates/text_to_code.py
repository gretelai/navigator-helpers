from dataclasses import dataclass, fields


@dataclass
class TextToCodePromptTemplates:

    domains_str: str = """\
Create a list of {num_domains} unique industries where you expect to find software engineers who code in {lang}. 

### Instructions:
    * Do not use abbreviations.
    * Keep each industry name to 1-5 words, preferring concise names.
    * List the industries in a valid JSON array.
"""

    topics_from_domains_str: str = """\
Create a list of {num_topics} topics that are associated with the following software domain: {domain}


### Instructions:
    * Do not use abbreviations.
    * Keep each topic name to 1-5 words, preferring concise names
    * List the topics in a valid JSON array.
"""

    complexity_str: str = """\
Come up with a list of {num_levels} complexity levels for software in the {lang} programming language.

### Instructions:
    * Each complexity level should be a short description of the level of complexity.
    * Do not mention specific libraries or frameworks.
    * List the levels in a valid JSON array.
    
#### Example:
    '["Beginner: Basic syntax and data types", "Intermediate: Functions", "Advanced: Object-oriented programming"]'
"""

    python_dependency_list_str: str = """\
Create the contents of a Python requirements.txt file with dependencies for a {project_type} project.

### Instructions:
    * Do not include package version numbers.
    * Do not include any comments.
    * Limit the number of dependencies to the most common ones.
    * Do not exceed {max_dependencies} dependencies.
    * Enclose the package names in triple backticks.

### Example:

```
package1
package2
package3
```
"""

    text_to_code_prompt_str: str = """\
Generate a natural language prompt that describes a {lang} coding task.

### Instructions:
    * Use a code complexity of "{complexity}".
    * Imagine the code will be used in the "{domain}" domain.
    * Write code that might be used in a "{topic}" context.
    * Return only the prompt without any code or other comments.
    
### Prompt:
"""

    generate_code_str: str = """\
{text_to_code_prompt}

### Instructions
    * The code should have a complexity of "{complexity}".
    * Write code that might be used in the "{domain}" industry within a "{topic}" context.
    * Try to include at least 1 of the following Python packages: {dependency_string}.
    * Include only the code, without any comments or additional text.
"""

    def __call__(self, template_name, **kwargs):
        return getattr(self, template_name).format(**kwargs)

    def __post_init__(self):
        """Add helper methods for easy access to the `format` method of each template.

        Usage::
            templates = TextToCodePromptTemplates()
            templates.domains(num_domains=10, lang="Python")
        """
        #
        for field in fields(self):
            setattr(
                self, field.name.replace("_str", ""), getattr(self, field.name).format
            )
