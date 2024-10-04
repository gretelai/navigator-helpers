DATASEED_REVERSE_ENG_PROMPT_TEMPLATE = """
You are an expert data practioner, skilled in critically evaluating data and
leveraging expertise across various domains, including data analysis, data
engineering, and a general understanding of programming concepts.

Examine the dataset provided in JSONL format inside the <dataset></dataset> tags
below. Note the schema of the dataset provided in the <schema></schema> tags.

Then reverse-engineer the dataset and think through which specific data seeds
could be used if such a dataset were to be generated from scratch so that the
data is rich, diverse and high-quality.

Here are a few examples to better explain what data seeds are. Don't use these
in your output
<examples>
  <example_1> A product review dataset with 'product_name' and 'review_text' columns could have the following columns as data seeds:
    - 'category' (representing categories like electronics, clothing, food, etc.)
    - 'reviewer_demographics' (representing factors like age, gender, location)
    - 'sentiment' (representing sentiments such as positive, neutral, negative)
    - 'product_price_range' (representing price ranges like low, medium, high)
    - 'purchase_channel' (representing online store, physical store, mobile app)
    - 'review_length' (representing the length of the review: short, medium, long)
  </example_1>
  <example_2>
    A dataset for environmental sensor readings with 'timestamp' and 'reading_value' columns could have the following columns as data seeds:
    - 'sensor_type' (representing things like temperature, humidity, CO2 levels, etc.)
    - 'location' (representing different geographical locations or zones)
    - 'weather_conditions' (representing variables like sunny, rainy, cloudy)
    - 'altitude' (representing different altitude ranges, such as sea level, mid-altitude, high-altitude)
    - 'sensor_brand' (representing brands or manufacturers of sensors)
    - 'data_granularity' (representing frequency of data collection, such as hourly, daily, or monthly)
  </example_2>
</examples

Provide data seeds as new columns in a dataset. DO NOT include columns from the
original dataset provided in <schema></schema> tags.
New column names should be succinct and descriptive.
DO NOT reference generic terms like seed, data in new column names.
DO NOT include any answers or questions in data seeds.
MAKE SURE TO USE SNAKE CASE IN COLUMN NAMES.

Return enhanced schema in the JSON_OUTPUT_FORMAT format specified below:

#### JSON_OUTPUT_FORMAT
{{
  "columns":
    [
        {{
            "column_name": "name_of_column_in_snake_case",
            "description": "data description",
            "example_values": [examples of the data seed provided as a python list]
        }},
        {{
            "column_name": "name_of_column_in_snake_case",
            "description": "data description",
            "example_values": [examples of the data seed provided as a python list]
        }}
    ]
}}

<dataset>
{sampled_dataset_jsonl}
</dataset>

<schema>
{sampled_dataset_column_list}
</schema>

This is the output format you have to follow:
<json>
  JSON goes here
</json>
"""


DATASEED_CROWD_RANKING_PROMPT_TEMPLATE = """
You are an expert data practioner, skilled in critically evaluating data and
leveraging expertise across various domains, including data analysis, data
engineering, and a general understanding of programming concepts.

Examine the dataset provided in JSONL format inside the <dataset></dataset> tags
below. Note the schema of the dataset in the <schema></schema> tags.

<dataset>
{sampled_dataset_jsonl}
</dataset>
<schema>
{sampled_dataset_column_list}
</schema>

Then examine and compare ALL data seeds inside the <data_seeds></data_seeds>
tags that could be used if such a dataset were to be generated from scratch
so that the data is rich, diverse and high-quality.
<data_seeds>
{data_seeds}
</data_seeds>

Based on the comparison, create a deduplicated and ranked list of data seeds. First,
make sure to prune the list to remove duplicate, similar and/or irrelevant data seeds.
Second, rank the ramining list based on relevance,
clarity, and diversity of the attributes, as they pertain to the dataset.
Use a three-point scale from 1-3 (1 for low, 2 for medium, 3 for high quality).
Return the data seed list, starting from highest to lowest quality.
Follow the JSON format below. DO NOT write any code to perform the ranking.

#### JSON FORMAT TO FOLLOW
{{
  "columns":
    [
        {{
            "column_name": "name_of_column_in_snake_case",
            "description": "data description",
            "example_values": [examples of the data seed provided as a python list],
            "quality_rank": "rank of the attribute based on quality"
        }},
        {{
            "column_name": "name_of_column_in_snake_case",
            "description": "data description",
            "example_values": [examples of the data seed provided as a python list],
            "quality_rank": "rank of the attribute based on quality"
        }}
    ]
}}

This is the output format you have to follow:
<json>
  JSON goes here
</json>
"""

DATASEED_GENERATION_PROMPT_TEMPLATE = """
Examine column descriptions provided in JSON format inside the
<data_seeds></data_seeds> tags.

Then generate a diverse, rich, and relevant list of values
for each column and add that list as a new (key,value) pair inside JSON.
Use all_examples as the key. Provide values as a list.
DO NOT use "etc." as a value.

<data_seeds>
{data_seeds}
</data_seeds>

Follow the JSON format below. DO NOT write any code to perform the ranking.

#### JSON FORMAT TO FOLLOW
{{
"columns":
    [
        {{
            "column_name": "name_of_column_in_snake_case",
            "description": "data description",
            "example_values": [examples of the data seed provided as a python list],
            "quality_rank": "rank of the attribute based on quality",
            "all_values": [generated diverse, rich, relevant values provided as a python list]
            ""
        }},
        {{
            "column_name": "name of column",
            "description": "data description",
            "example_values": [examples of the data seed provided as a python list],
            "quality_rank": "rank of the attribute based on quality",
            "all_values": [generated diverse, rich, relevant values provided as a python list]
        }}
    ]
}}

This is the output format you have to follow:
<json>
JSON goes here
</json>
"""


DATASET_DESCRIPTION_PROMPT_TEMPLATE = """
You are an expert data practioner, skilled in critically evaluating data and
leveraging expertise across various domains, including data analysis, data
engineering, and a general understanding of programming concepts.

Carefuly Examine the dataset provided in JSONL format inside the <dataset></dataset> tags
below. Note the schema of the dataset provided in he <schema></schema> tags.

<dataset>
{sampled_dataset_jsonl}
</dataset>
<schema>
{sampled_dataset_column_list}
</schema>

Provide a comprehensive description of the dataset, highlighting it's focus,
structure, formatting, and the most likely use-case and/or intent.

You MUST clearly state the name of each column in the dataset in the description.
You MUST provide a brief description for that column.

Follow the exact JSON format below. Do not add any other keys to JSON.
DO NOT write any code to perform the ranking.

#### JSON FORMAT TO FOLLOW
{{
  "dataset_description": {{
    "description": "The dataset represents/captures/is ...",
    "columns": [
      {{
        "column_name": "name_of_column_in_snake_case",
        "description": "brief column description"
      }},
      {{
        "column_name": "name_of_column_in_snake_case",
        "description": "brief column description"
      }}
    ]
  }}
}}

This is the output format you have to follow:
<json>
  JSON goes here
</json>

Make sure to reread the above to fully understand the instructions.
"""

DATA_GENERATION_PROMPT_TEMPLATE = """
You are an expert data practioner, skilled in critically evaluating data and
leveraging expertise across various domains, including data analysis, data
engineering, and a general understanding of programming concepts.

Examine the dataset description in <dataset_description> tags.
<dataset_description>
{dataset_description}
</dataset_description>

Turn this description into an LLM prompt that would ask an AI assistant to
create such a dataset.

Then, augment your prompt with dataseeds provided in <data_seeds></data_seeds>
tags by asking to consider them as important context for data.
DO NOT use the phrase "data seeds" in the prompt.
DO NOT include data seeds as additional columns in the dataset.
DO NOT include any other data seeds.
DO NOT use the JSON format in your prompt.
DO NOT write code for data seed inclusion.
<data_seeds>
{data_seeds}
</data_seeds>

Here are a few examples to better explain how to use data seeds in the prompt.
Don't use these examples in your output.
<examples>
  <example_1>
    A product review dataset with 'product_name' and 'review_text'
    columns could have 'category' and 'sentiment' as data seeds. A prompt could
    look like this:
    - Generate a rich and diverse product review dataset with two columns:
    'product_name' and 'review_text'. Focus on the {{category}} category of
    products and make sure to express then {{sentiment}} sentiment.
  </example_1>
  <example_2>
    A dataset for environmental sensor readings with 'timestamp' and
    'reading_value' columns could have 'sensor_type', 'sensor_brand' and
    'weather_conditions' as data seeds. A prompt could look like this:
    - Create a well-structured dataset of environmental sensor readings that has
    the following two columns in it: timestamp, reading_value. The dataset
    should represent readings from {{sensor_type}} type sensors produced
    by {{sensor_brand}} in {{weather_conditions}} conditions.
  </example_2>
</examples

Follow the exact JSON format below. Do not add any other keys to JSON.
DO NOT write any code to generate the prompt

#### JSON FORMAT TO FOLLOW
{{
  "prompt": "generated prompt"
}}

This is the output format you have to follow:
<json>
  JSON goes here
</json>
"""

JSONL_DATA_GENERATION_PROMPT_TEMPLATE = """
{data_generation_prompt}

Generate {num_records} records of data in JSONL format.
DO NOT WRITE ANY CODE TO GENERATE DATA. RETURN JUST DATA ITSELF.
Make sure the data in each record adheres to the format in examples.

This is the output format you have to follow:
<json>
  [
    {{row 1 of the dataset}},
    {{row 2 of the dataset}}
  ]
</json>

Reread the instructions before proceeding, paying attention to format
in examples: {data_generation_prompt}
"""