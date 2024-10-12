pii_document_template_dict = dict(
    domains="""\
Create a list of {num_domains} unique industry sector where you expect to find documents that containing PII and PHI related information. 

### Instructions:
    * Do not use abbreviations.
    * Keep each industry name to 1-5 words, preferring concise names.
    * List the industries in a valid JSON array.
""",
    doctypes_from_domains="""\
You are a data expert specializing in the domain of {domain}. Your task is to create a comprehensive list of {num_doctypes} document types relevant to {domain}, focusing on formats and schemas as they relate to the customer, user or author journey within creating documents applicable to {domain}.

### Instructions:
    * Do not use abbreviations.
    * Keep each document type to 1-5 words, preferring concise names
    * List the document types in a valid JSON array.
""",
    document_description_generation="""\
Write a one-sentence detailed description of the document of the following domain and document type: {domain} and {doctype} Include specifics about the document's format, common fields, and content type where applicable.
""",
    pii_type_generation = """\
Create a list of {num_pii_types} unique PII and PHI type entities that are relevant for a document described as: {docdesc}

### Instructions:
    * Only selected PII types from the following list and use the exact same naming: {allowed_pii_types}
    * Ensure a balanced mix of both common and uncommon PII types from the list. Prioritize relevance but aim to include some less frequently used types as well.
    * Where applicable, include state or country information.
    * Do not make up any PII type names beyond those provided in the list.
    * Do not add any special characters such as line breaks or spaces.
    * List the PII types in a valid JSON array.
""",
    document_generation="""\
You are an expert language model trained in various domains, capable of generating accurate and contextually relevant text passages. Your task is to generate a text passage based on the following instructions.

### Document Details:
- **Domain**: {domain}
- **Document Type**: {doctype}
- **Description**: {docdesc}

### Entities to Include:
{entities}

### Instructions:
1. **Text Generation**: Generate a realistic and contextually appropriate text passage that could appear in a document of type `{doctype}` within the domain of `{domain}`. The document is described as follows: `{docdesc}`.
2. **Incorporate Entities Exactly as Provided**: Ensure that the exact PII values specified in the entities field are included verbatim in the generated text. No changes to format, spelling, capitalization, or structure of the PII values should be made. For instance, if the entity is date_of_birth: 1925-05-06, the text must use 1925-05-06 exactly as given, not a reformatted version like May 6, 1925.
3. **Exclude Unlisted Entities**: Do **not** include any entities or information other than those explicitly listed in the `entities_json` field. The text should only reflect the entities provided.
4. **Consistency and Accuracy**: The PII values should fit naturally into the context of the document but remain exactly as provided in the entities list.

### Important:
- **Output Only the Text**: **Do not** include any additional JSON structure, explanations, lists, or annotations. The output should be **solely** the generated text content that would appear in the document.

### Example Expected Output:
"This Lease Agreement is entered into on 01/01/2023 between Cynthia Long-Garza, the tenant, born on 1925-05-06, and the property owner, for the rental of the property located at 4 Berry mountains, Apt. 72."
""",
)


pii_code_template_dicts = {
    "pii_doc": pii_document_template_dict,
}
