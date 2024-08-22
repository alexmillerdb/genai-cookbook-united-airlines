# Databricks notebook source
# MAGIC %pip install -U -qqqq pyyaml mlflow mlflow-skinny

# COMMAND ----------

# MAGIC %md # POC configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ## Application configuration
# MAGIC
# MAGIC To begin with, we simply need to configure the following:
# MAGIC 1. `RAG_APP_NAME`: The name of the RAG application.  Used to name the app's Unity Catalog model and is prepended to the output Delta Tables + Vector Indexes
# MAGIC 2. `UC_CATALOG` & `UC_SCHEMA`: [Create a Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/create-catalogs.html#create-a-catalog) and a Schema where the output Delta Tables with the parsed/chunked documents and Vector Search indexes are stored
# MAGIC 3. `UC_MODEL_NAME`: Unity Catalog location to log and store the chain's model
# MAGIC 4. `VECTOR_SEARCH_ENDPOINT`: [Create a Vector Search Endpoint](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-endpoint) to host the resulting vector index
# MAGIC 5. `SOURCE_PATH`: A [UC Volume](https://docs.databricks.com/en/connect/unity-catalog/volumes.html#create-and-work-with-volumes) that contains the source documents for your application.
# MAGIC 6. `MLFLOW_EXPERIMENT_NAME`: MLflow Experiment to track all experiments for this application.  Using the same experiment allows you to track runs across Notebooks and have unified lineage and governance for your application.
# MAGIC 7. `EVALUATION_SET_FQN`: Delta Table where your evaluation set will be stored.  In the POC, we will seed the evaluation set with feedback you collect from your stakeholders.
# MAGIC
# MAGIC After finalizing your configuration, optionally run `01_validate_config` to check that all locations exist.

# COMMAND ----------

# By default, will use the current user name to create a unique UC catalog/schema & vector search endpoint
user_email = spark.sql("SELECT current_user() as username").collect()[0].username
user_name = user_email.split("@")[0].replace(".", "").lower()[:35]

dbutils.widgets.text(name="RAG_APP_NAME", defaultValue="united_airlines_rag_app", label="Application Name (must be unique)")
dbutils.widgets.text(name="UC_CATALOG", defaultValue="main", label="UC Catalog Name")
dbutils.widgets.text(name="UC_SCHEMA", defaultValue="rag_united_airlines", label="UC Schema (must be unique)")
dbutils.widgets.text(name="VECTOR_SEARCH_ENDPOINT", defaultValue="one-env-shared-endpoint-0", label="VS Endpoint (do not update)")

# The name of the RAG application.  This is used to name the chain's UC model and prepended to the output Delta Tables + Vector Indexes
RAG_APP_NAME = dbutils.widgets.get("RAG_APP_NAME")

# UC Catalog & Schema where outputs tables/indexs are saved
# If this catalog/schema does not exist, you need create catalog/schema permissions.
UC_CATALOG = dbutils.widgets.get("UC_CATALOG")
UC_SCHEMA = dbutils.widgets.get("UC_SCHEMA")

## UC Model name where the POC chain is logged
UC_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{RAG_APP_NAME}"

# Vector Search endpoint where index is loaded
# If this does not exist, it will be created
VECTOR_SEARCH_ENDPOINT = dbutils.widgets.get("VECTOR_SEARCH_ENDPOINT")

# Source location for documents
# You need to create this location and add files
SOURCE_PATH = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/source_docs"

############################
##### We suggest accepting these defaults unless you need to change them. ######
############################

EVALUATION_SET_FQN = f"`{UC_CATALOG}`.`{UC_SCHEMA}`.{RAG_APP_NAME}_evaluation_set"

# MLflow experiment name
# Using the same MLflow experiment for a single app allows you to compare runs across Notebooks
import mlflow
MLFLOW_EXPERIMENT_NAME = f"/Users/{user_email}/{RAG_APP_NAME}"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# MLflow Run Names
# These Runs will store your initial POC application.  They are later used to evaluate the POC model against your experiments to improve quality.

# Data pipeline MLflow run name
POC_DATA_PIPELINE_RUN_NAME = f"data_pipeline_{RAG_APP_NAME}"
# Chain MLflow run name
POC_CHAIN_RUN_NAME = f"chain_{RAG_APP_NAME}"

# COMMAND ----------

print(f"RAG_APP_NAME {RAG_APP_NAME}")
print(f"UC_CATALOG {UC_CATALOG}")
print(f"UC_SCHEMA {UC_SCHEMA}")
print(f"UC_MODEL_NAME {UC_MODEL_NAME}")
print(f"VECTOR_SEARCH_ENDPOINT {VECTOR_SEARCH_ENDPOINT}")
print(f"SOURCE_PATH {SOURCE_PATH}")
print(f"EVALUATION_SET_FQN {EVALUATION_SET_FQN}")
print(f"MLFLOW_EXPERIMENT_NAME {MLFLOW_EXPERIMENT_NAME}")
print(f"POC_DATA_PIPELINE_RUN_NAME {POC_DATA_PIPELINE_RUN_NAME}")
print(f"POC_CHAIN_RUN_NAME {POC_CHAIN_RUN_NAME}")

# COMMAND ----------

print(f"POC app using the UC catalog/schema {UC_CATALOG}.{UC_SCHEMA} with source data from {SOURCE_PATH} synced to the Vector Search endpoint {VECTOR_SEARCH_ENDPOINT}.  \n\nChain model will be logged to UC as {UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}.  \n\nUsing MLflow Experiment `{MLFLOW_EXPERIMENT_NAME}` with data pipeline run name `{POC_DATA_PIPELINE_RUN_NAME}` and chain run name `{POC_CHAIN_RUN_NAME}`")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # POC Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data preparation
# MAGIC ### Config
# MAGIC
# MAGIC Databricks reccomends starting with the default settings below for your POC.  Once you have collected stakeholder feedback, you will iterate on the app's quality using these parameters.
# MAGIC
# MAGIC To learn more about these settings, visit [link to guide].
# MAGIC
# MAGIC By default, we use [GTE Embeddings](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#call-a-bge-embeddings-model-using-databricks-model-serving-notebook) that is available on [Databricks Foundation Model APIs](https://docs.databricks.com/en/machine-learning/foundation-models/index.html).  GTE is a high quality open source embedding model with a large context window.  We have selected a tokenizer and chunk size that matches this embedding model.

# COMMAND ----------

dbutils.widgets.text("embedding_endpoint_name", 
                     defaultValue="databricks-gte-large-en",
                     label="Embedding endpoint name")

embedding_endpoint_name = dbutils.widgets.get("embedding_endpoint_name")

if "gte-large-en" in embedding_endpoint_name:
  tokenizer_model_name = "Alibaba-NLP/gte-large-en-v1.5"
  tokenizer_source = "hugging_face"
elif "bge-large-en" in embedding_endpoint_name:
  tokenizer_model_name = "BAAI/bge-large-en-v1.5"
  tokenizer_source = "hugging_face"
else:
  # tokenizer_model_name = "cl100k_base"
  tokenizer_model_name = ""
  tokenizer_source = ""


print(f"Embedding endpoint name: {embedding_endpoint_name}")
print(f"Tokenizer model name: {tokenizer_model_name}")

# COMMAND ----------

# dbutils.widgets.text("chunk_size_tokens", "1024", label="Chunk size for documents")
# dbutils.widgets.text("chunk_overlap_tokens", "256", label="Chunk size for overlap of documents")

# chunk_size_tokens = int(dbutils.widgets.get("chunk_size_tokens"))
# chunk_overlap_tokens = int(dbutils.widgets.get("chunk_overlap_tokens"))
chunk_size_tokens = 1024
chunk_overlap_tokens = 256

print(f"Chunk size {chunk_size_tokens} with chunk overlap of {chunk_overlap_tokens}")

# COMMAND ----------

data_pipeline_config = {
    # Vector Search index configuration
    "vectorsearch_config": {
        # Pipeline execution mode.
        # TRIGGERED: If the pipeline uses the triggered execution mode, the system stops processing after successfully refreshing the source table in the pipeline once, ensuring the table is updated based on the data available when the update started.
        # CONTINUOUS: If the pipeline uses continuous execution, the pipeline processes new data as it arrives in the source table to keep vector index fresh.
        "pipeline_type": "TRIGGERED",
    },
    # Embedding model to use
    # Tested configurations are available in the `supported_configs/embedding_models` Notebook
    "embedding_config": {
        # Model Serving endpoint name
        "embedding_endpoint_name": embedding_endpoint_name,
        "embedding_tokenizer": {
            # Name of the embedding model that the tokenizer recognizes
            "tokenizer_model_name": tokenizer_model_name,
            # Name of the tokenizer, either `hugging_face` or `tiktoken`
            "tokenizer_source": tokenizer_source,
        },
    },
    # Parsing and chunking configuration
    # Changing this configuration here will NOT impact your data pipeline, these values are hardcoded in the POC data pipeline.
    # It is provided so you can copy / paste this configuration directly into the `Improve RAG quality` step and replicate the POC's data pipeline configuration
    "pipeline_config": {
        # File format of the source documents
        "file_format": "pdf",
        # Parser to use (must be present in `parser_library` Notebook)
        "parser": {"name": "pypdf", "config": {}},
        # Chunker to use (must be present in `chunker_library` Notebook)
        "chunker": {
            "name": "langchain_recursive_char",
            "config": {
                "chunk_size_tokens": chunk_size_tokens,
                "chunk_overlap_tokens": chunk_overlap_tokens,
            },
        },
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Output tables
# MAGIC
# MAGIC Next, we configure the output Delta Tables and Vector Index where the data pipeline will write the parsed/chunked/embedded data.

# COMMAND ----------

# Names of the output Delta Tables tables & Vector Search index
destination_tables_config = {
    # Staging table with the raw files & metadata
    "raw_files_table_name": f"`{UC_CATALOG}`.`{UC_SCHEMA}`.`{RAG_APP_NAME}_poc_raw_files_bronze`",
    # Parsed documents
    "parsed_docs_table_name": f"`{UC_CATALOG}`.`{UC_SCHEMA}`.`{RAG_APP_NAME}_poc_parsed_docs_silver`",
    # Chunked documents that are loaded into the Vector Index
    "chunked_docs_table_name": f"`{UC_CATALOG}`.`{UC_SCHEMA}`.`{RAG_APP_NAME}_poc_chunked_docs_gold`",
    # Destination Vector Index
    "vectorsearch_index_table_name": f"`{UC_CATALOG}`.`{UC_SCHEMA}`.`{RAG_APP_NAME}_poc_chunked_docs_gold_index`",
}
destination_tables_config["vectorsearch_index_name"] = destination_tables_config["vectorsearch_index_table_name"].replace("`", "")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load config
# MAGIC
# MAGIC This step loads the configuration so that the `02_poc_data_preperation` Notebook can use it.

# COMMAND ----------

import json

vectorsearch_config = data_pipeline_config['vectorsearch_config']
embedding_config = data_pipeline_config['embedding_config']
pipeline_config = data_pipeline_config['pipeline_config']

print(f"Using POC data pipeline config: {json.dumps(data_pipeline_config, indent=4)}\n")
print(f"Writing to: {json.dumps(destination_tables_config, indent=4)}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Chain config
# MAGIC
# MAGIC Next, we configure the chain's default settings.  The chain's code has been parameterized to use these variables. 
# MAGIC
# MAGIC Again, Databricks reccomends starting with the default settings below for your POC.  Once you have collected stakeholder feedback, you will iterate on the app's quality using these parameters.
# MAGIC
# MAGIC By default, we use `databricks-dbrx-instruct` but you can change this to any LLM hosted using Databricks Model Serving, including Azure OpenAI / OpenAI models.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Single or multi turn conversation?
# MAGIC
# MAGIC Let's take a sample converastion:
# MAGIC
# MAGIC > User: What is RAG?<br/>
# MAGIC > Assistant: RAG is a technique...<br/>
# MAGIC > User: How do I do it?<br/>
# MAGIC
# MAGIC A multi-turn conversation chain allows the assistant to understand that *it* in *how do I do it?* refers to *RAG*.  A single-turn conversation chain would not understand this context, as it treats every request as a completely new question.
# MAGIC
# MAGIC Most RAG use cases are for multi-turn conversation, however, the additional step required to understand *how do I do it?* uses an LLM and thus adds a small amount of latency.

# COMMAND ----------

# Notebook with the chain's code.  Choose one based on your requirements.  
# If you are not sure, use the `multi_turn_rag_chain`.

# CHAIN_CODE_FILE = "single_turn_rag_chain"

CHAIN_CODE_FILE = "multi_turn_rag_chain"

# COMMAND ----------

dbutils.widgets.text("llm_endpoint_name", 
                     defaultValue="databricks-meta-llama-3-1-70b-instruct",
                     label="LLM Endpoint Name")

llm_endpoint_name = dbutils.widgets.get("llm_endpoint_name")

dbutils.widgets.text("llm_system_prompt_template", 
                     defaultValue="""You are an assistant that answers questions. Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.
                     
                     Context: {context}""",
                     label="LLM System Prompt for application") 

llm_system_prompt_template = dbutils.widgets.get("llm_system_prompt_template")

# COMMAND ----------

# Chain configuration
# We suggest using these default settings
rag_chain_config = {
    "databricks_resources": {
        # Only required if using Databricks vector search
        "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT,
        # Databricks Model Serving endpoint name
        # This is the generator LLM where your LLM queries are sent.
        "llm_endpoint_name": llm_endpoint_name,
    },
    "retriever_config": {
        # Vector Search index that is created by the data pipeline
        "vector_search_index": destination_tables_config["vectorsearch_index_name"],
        "embedding_endpoint_name": embedding_endpoint_name,
        # "embedding_dimension": 1024,
        "schema": {
            # The column name in the retriever's response referred to the unique key
            # If using Databricks vector search with delta sync, this should the column of the delta table that acts as the primary key
            "primary_key": "chunk_id",
            # The column name in the retriever's response that contains the returned chunk.
            "chunk_text": "chunked_text",
            # The template of the chunk returned by the retriever - used to format the chunk for presentation to the LLM.
            "document_uri": "path",
        },
        # Prompt template used to format the retrieved information to present to the LLM to help in answering the user's question
        "chunk_template": "Passage: {chunk_text}\n",
        # The column name in the retriever's response that refers to the original document.
        "parameters": {
            # Number of search results that the retriever returns
            "k": 5,
            # Type of search to run
            # Semantic search: `ann`
            # Hybrid search (keyword + sementic search): `hybrid`
            "query_type": "ann",
        },
        # Tag for the data pipeline, allowing you to easily compare the POC results vs. future data pipeline configurations you try.
        "data_pipeline_tag": "poc",
    },
    "llm_config": {
        # Define a template for the LLM prompt.  This is how the RAG chain combines the user's question and the retrieved context.
        "llm_system_prompt_template": llm_system_prompt_template.strip(),
        # Parameters that control how the LLM responds.
        "llm_parameters": {"temperature": 0.01, "max_tokens": 1500},
    },
    "input_example": {
        "messages": [
            {
                "role": "user",
                "content": "What is Baggage Claim ETA?",
            },
        ]
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load config & save to YAML
# MAGIC
# MAGIC This step saves the configuration so that the `03_deploy_poc_to_review_app` Notebook can use it.

# COMMAND ----------

import yaml
print(f"Using chain config: {json.dumps(rag_chain_config, indent=4)}\n\n Using chain file: {CHAIN_CODE_FILE}")

with open('rag_chain_config.yaml', 'w') as f:
    yaml.dump(rag_chain_config, f)

# COMMAND ----------

# MAGIC %md ## Load shared utilities used by the other notebooks

# COMMAND ----------

# MAGIC %run ../z_shared_utilities
