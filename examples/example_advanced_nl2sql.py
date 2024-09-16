import textwrap

from typing import Dict, List

import pandas as pd

from navigator_helpers import (
    batch_and_write_data,
    DataFieldDefinition,
    DataModelDefinition,
    EvolDataGenerator,
    GeneratorConfig,
    mix_contextual_tags,
)

# Constants
NUM_TAGS = 100000  # Total number of synthetic records to generate
BATCH_SIZE = 20  # Save to disk at this frequency


def get_contextual_dataframes() -> List[pd.DataFrame]:
    """Returns a list of dataframes used for contextual tags."""

    df_sql_complexity = pd.DataFrame(
        {
            "complexity": ["Moderate", "Complex", "Very Complex", "Expert"],
            "complexity_description": [
                "Requires 2-3 SQL operations to solve the problem",
                "Requires 4-5 SQL operations and intermediate result handling",
                "Requires 6+ SQL operations, subqueries, and complex logic",
                "Requires advanced SQL features, recursive queries, or dynamic SQL",
            ],
        }
    )

    df_sql_concepts = pd.DataFrame(
        {
            "concept": [
                "Multi-table Joins",
                "Subqueries",
                "Window Functions",
                "Common Table Expressions (CTEs)",
                "Set Operations",
                "Advanced Aggregations",
                "Recursive Queries",
                "Pivoting and Unpivoting",
                "Date and Time Operations",
                "String Manipulations",
                "Conditional Logic",
                "Data Type Conversions",
                "Hierarchical Data Queries",
                "Complex Grouping",
                "Advanced Filtering",
                "JSON Operations",
                "Full-Text Search",
                "Geospatial Queries",
                "Temporal Data Handling",
                "Dynamic SQL Generation",
            ],
            "weight": [
                18,
                18,
                15,
                15,
                10,
                12,
                8,
                10,
                15,
                15,
                15,
                12,
                7,
                10,
                15,
                10,
                7,
                7,
                10,
                8,
            ],
        }
    )

    df_domains = pd.DataFrame(
        {
            "domain": [
                "Healthcare",
                "Finance",
                "E-commerce",
                "Retail",
                "Manufacturing",
                "Telecommunications",
                "Insurance",
                "Education",
                "Real Estate",
                "Transportation and Logistics",
                "Energy and Utilities",
                "Media and Entertainment",
                "Government",
                "Nonprofit",
                "Agriculture",
                "Hospitality",
                "IoT",
                "Gaming",
                "Social Media",
                "Cybersecurity",
            ],
            "description": [
                "Managing patient records, clinical data, and healthcare operations.",
                "Banking, investment, and financial management systems.",
                "Online stores, shopping carts, and transaction management.",
                "Physical stores and supply chain management for retail operations.",
                "Production processes, inventory, and automation in factories.",
                "Mobile networks, internet services, and communication technologies.",
                "Risk assessment, claims, and policy management in insurance.",
                "Learning management systems and student data in educational institutions.",
                "Property listings, management, and real estate transactions.",
                "Managing supply chain, logistics, and transportation systems.",
                "Power generation, utility management, and resource optimization.",
                "Content creation, streaming services, and audience engagement.",
                "Public sector operations, citizen services, and governmental data.",
                "Charitable organizations and their management of resources and services.",
                "Farming, livestock, and crop management for agricultural operations.",
                "Hospital and tourism operations, bookings, and customer service.",
                "Internet of Things devices and their data management.",
                "Video game development, player analytics, and real-time processing.",
                "User-generated content, engagement, and platform management.",
                "Data security, intrusion detection, and threat monitoring.",
            ],
        }
    )

    df_data_structures = pd.DataFrame(
        {
            "data_structure": [
                "Relational Tables",
                "JSON Documents",
                "XML Documents",
                "Key-Value Pairs",
                "Time Series",
                "Graph Data",
                "Geospatial Data",
                "Hierarchical Data",
                "Document Store",
                "Column-Oriented",
            ],
            "weight": [100, 20, 10, 5, 15, 5, 10, 10, 5, 5],
        }
    )

    df_data_types = pd.DataFrame(
        {
            "data_type_combination": [
                "Integer, Varchar, Date, Timestamp",
                "Integer, Float, Varchar, Text, Timestamp",
                "Integer, Decimal, Varchar, Date",
                "Integer, Varchar, Text, JSON",
                "Integer, Varchar, Timestamp, UUID",
                "Integer, Varchar, Date, Boolean",
                "Integer, Float, Varchar, Geometry",
                "Integer, Varchar, XML, Timestamp",
                "Integer, Varchar, JSONB, Array",
                "Integer, Varchar, Date, Interval",
                "Integer, Varchar, Decimal, UUID",
                "Integer, Varchar, JSONB, Boolean",
                "Integer, Varchar, Date, ENUM",
                "Integer, Varchar, BLOB, Timestamp",
            ],
            "weight": [20, 20, 15, 12, 8, 10, 5, 3, 7, 2, 5, 6, 3, 2],
        }
    )

    df_analysis_types = pd.DataFrame(
        {
            "analysis_type": [
                "Trend Analysis",
                "Cohort Analysis",
                "Funnel Analysis",
                "Anomaly Detection",
                "Predictive Modeling",
                "Customer Segmentation",
                "A/B Testing",
                "Time Series Analysis",
                "Geospatial Analysis",
                "Network Analysis",
                "Sentiment Analysis",
                "Risk Assessment",
                "Resource Optimization",
                "Performance Benchmarking",
                "Impact Analysis",
                "Fraud Detection",
                "Inventory Optimization",
                "Churn Prediction",
                "Recommendation Systems",
                "User Behavior Analysis",
                "Supply Chain Optimization",
                "Credit Scoring",
                "Deep Learning-based Analysis",
                "Market Segmentation",
                "Operational Efficiency",
            ]
        }
    )

    return [
        df_sql_complexity,
        df_sql_concepts,
        df_domains,
        df_data_structures,
        df_data_types,
        df_analysis_types,
    ]


def create_model_definition() -> DataModelDefinition:
    """Creates and returns the DataModelDefinition for SQL scenarios."""
    return DataModelDefinition(
        generation_instructions=textwrap.dedent(
            """You are a seasoned SQL expert specializing in crafting intricate, context-rich queries and explanations. 
        * Use the provided contextual tags as instructions for generation. Reference the provided Context to ensure relevance and appropriate complexity.
        * Use proper spacing for all generated SQL content, with no extra comments or explanations when generating code.
        * All SQL data should be PostgreSQL compatible SQL.
        * Avoid excessive newlines, especially at the end of generated content.
        """
        ),
        fields=[
            DataFieldDefinition(
                name="sql_context",
                type="str",
                description="A single string comprising multiple valid PostgreSQL `CREATE TABLE` statements and a complex schema similar to a production application including multiple tables, separated by semicolons. The schema should be based on the provided Context, particularly the domain and domain_description.",
                validator="sql:postgres",
                evolution_strategies=[
                    "Enhance the schema to include domain-specific tables and data types.",
                    "Add relevant indexes, constraints, and views reflecting real-world designs.",
                ],
                evolution_rate=0.1,
            ),
            DataFieldDefinition(
                name="prompt",
                type="str",
                description="A detailed, nuanced natural language prompt that a user might ask to a database for a particular task, based on the provided `sql_context` field that challenges advanced understanding. The prompt should align with the domain and domain_description from the contextual tags.",
                validator="A natural language question or command written in English",
                evolution_strategies=[
                    "Refine the prompt to sound more natural.",
                    "Ensure the prompt reflects real-world business needs.",
                ],
                evolution_rate=0.1,
            ),
            DataFieldDefinition(
                name="sql",
                type="str",
                description="A fully executable SQL query that directly answers the `prompt` using the schema in `sql_context`, with no markup or extraneous explanations. The query complexity should match the sql_complexity specified in the contextual tags.",
                validator="sql:postgres",
                evolution_strategies=[
                    "Optimize the SQL solution for performance.",
                    "Ensure the solution handles large datasets efficiently.",
                ],
                evolution_rate=0.1,
            ),
            DataFieldDefinition(
                name="sql_explanation",
                type="str",
                description="A comprehensive step-by-step breakdown of the SQL query, detailing how it answers the `prompt` and the purpose of each part. Include references to the domain-specific context.",
                evolution_strategies=[
                    "Provide a clear, step-by-step explanation.",
                    "Explain the rationale behind specific SQL techniques.",
                ],
                evolution_rate=0.1,
            ),
        ],
    )


def main():
    """Main function to generate synthetic SQL scenarios."""
    print("Starting advanced SQL scenario generation...")

    # Create contextual tags
    contextual_tags = mix_contextual_tags(NUM_TAGS, get_contextual_dataframes())

    # Print overview of contextual tags
    print(contextual_tags.head(1).T)
    print(contextual_tags["data_structure"].value_counts(normalize=True))
    print(contextual_tags["concept"].value_counts(normalize=True))

    # Set up generator configuration
    config = GeneratorConfig(
        api_key="prompt",
        llm_model="gretelai/gpt-auto",
        num_generations=1,
        log_level="INFO",
        use_reflection=True,
    )

    # Initialize the synthetic data generator
    generator = EvolDataGenerator(
        config,
        create_model_definition(),
    )

    # Generate and write data in batches
    batch_and_write_data(
        generator=generator,
        contextual_tags=contextual_tags,
        batch_size=BATCH_SIZE,
        file_prefix="advanced_nl2sql",
    )

    print("Advanced SQL scenario generation complete.")


if __name__ == "__main__":
    main()
