# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-sdk
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md ## Update LLM System Prompt for your AI application. 
# MAGIC
# MAGIC A system prompt is a set of instructions that are given to LLM to guide the model's responses to user queries/questions. System prompts are critical part of how LLM's function and are included in every input, regardless of user's prompt. Developers use system prompts to limit the LLM's answer in a variety of ways such as:
# MAGIC - Defining the model's role: "You are an AI assistant that specializes in cuisine."
# MAGIC - Providing instructions: "Recommend recipes that the user might like based on their preferences."
# MAGIC - Constraining the answer: "Do not reply with anything harmful or inappropriate. The output should replicate a foodie."

# COMMAND ----------

# do not update Context: {context} in the prompt below
# this is important for the LLM to know that there will be context to help generate the answer
llm_system_prompt_template = """
    You are an assistant that answers questions. Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.
    
    Context: {context}
    """

# COMMAND ----------

# MAGIC %md TO DO:
# MAGIC - Add LLM Endpoint
# MAGIC - Add Embedding Endpoint
# MAGIC - Add Volume Path where they upload data and print out to upload data (create separate notebook to create data assets and validate endpoints are correct)
# MAGIC - Create compute/clusters (clone instructions)
# MAGIC - Update instructions

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

dbutils.widgets.text("llm_endpoint_name",
                     defaultValue="databricks-meta-llama-3-1-70b-instruct",
                     label="LLM Endpoint Name")

llm_endpoint_name = dbutils.widgets.get("llm_endpoint_name")

dbutils.widgets.text("embedding_endpoint_name",
                     defaultValue="databricks-gte-large-en",
                     label="Embedding endpoint name")

embedding_endpoint_name = dbutils.widgets.get("embedding_endpoint_name")

# COMMAND ----------

# MAGIC %md ### Create data assets for app

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointStatusState, EndpointType
from mlflow.utils import databricks_utils as du
from databricks.sdk.service.serving import EndpointCoreConfigInput, EndpointStateReady
from databricks.sdk.errors import ResourceDoesNotExist
import os

w = WorkspaceClient()
browser_url = du.get_browser_hostname()

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, PermissionDenied
w = WorkspaceClient()

# Create UC Catalog if it does not exist, otherwise, raise an exception
try:
    _ = w.catalogs.get(UC_CATALOG)
    print(f"PASS: UC catalog `{UC_CATALOG}` exists")
except NotFound as e:
    print(f"`{UC_CATALOG}` does not exist, trying to create...")
    try:
        _ = w.catalogs.create(name=UC_CATALOG)
    except PermissionDenied as e:
        print(f"FAIL: `{UC_CATALOG}` does not exist, and no permissions to create.  Please provide an existing UC Catalog.")
        raise ValueError(f"Unity Catalog `{UC_CATALOG}` does not exist.")
        
# Create UC Schema if it does not exist, otherwise, raise an exception
try:
    _ = w.schemas.get(full_name=f"{UC_CATALOG}.{UC_SCHEMA}")
    print(f"PASS: UC schema `{UC_CATALOG}.{UC_SCHEMA}` exists")
except NotFound as e:
    print(f"`{UC_CATALOG}.{UC_SCHEMA}` does not exist, trying to create...")
    try:
        _ = w.schemas.create(name=UC_SCHEMA, catalog_name=UC_CATALOG)
    except PermissionDenied as e:
        print(f"FAIL: `{UC_CATALOG}.{UC_SCHEMA}` does not exist, and no permissions to create.  Please provide an existing UC Schema.")
        raise ValueError("Unity Catalog Schema `{UC_CATALOG}.{UC_SCHEMA}` does not exist.")

# Check if source location exists
import os

if os.path.isdir(SOURCE_PATH):
    print(f"PASS: `{SOURCE_PATH}` exists")
else:
    print(f"`{SOURCE_PATH}` does NOT exist, trying to create")

    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service import catalog
    from databricks.sdk.errors import ResourceAlreadyExists

    w = WorkspaceClient()

    volume_name = SOURCE_PATH[9:].split('/')[2]
    uc_catalog = SOURCE_PATH[9:].split('/')[0]
    uc_schema = SOURCE_PATH[9:].split('/')[1]
    try:
        created_volume = w.volumes.create(
            catalog_name=uc_catalog,
            schema_name=uc_schema,
            name=volume_name,
            volume_type=catalog.VolumeType.MANAGED,
        )
        print(f"PASS: Created `{SOURCE_PATH}`")
    except Exception as e:
        print(f"`FAIL: {SOURCE_PATH}` does NOT exist, could not create due to {e}")
        raise ValueError("Please verify that `{SOURCE_PATH}` is a valid UC Volume")

print(f"All tests have passed. Upload your source PDF documents to {SOURCE_PATH}")

# COMMAND ----------

# MAGIC %md ## Build Job workflow that will run automatically run scripts

# COMMAND ----------

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import Task, NotebookTask, Source, TaskDependency

w = WorkspaceClient()

job_name = f"{RAG_APP_NAME}_workflow"
cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
databricks_url = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"
tasks_list = ['00_config', '01_validate_config', '02_poc_data_pipeline','03_deploy_poc_to_review_app']
current_directory = os.getcwd()

# checks if job has been created
jobs = w.jobs.list()
job_id = None

for job in jobs:
    if job.settings.name == job_name and job.creator_user_name == user_email:
        job_id = job.job_id
        break

# dynamically create tasks
tasks = []
for i, task in enumerate(tasks_list):
    if task == "00_config":
      base_parameters = {
        "RAG_APP_NAME": RAG_APP_NAME,
        "UC_CATALOG": UC_CATALOG,
        "UC_SCHEMA": UC_SCHEMA,
        "llm_system_prompt_template": llm_system_prompt_template,
        "llm_endpoint_name": llm_endpoint_name,
        "VECTOR_SEARCH_ENDPOINT": VECTOR_SEARCH_ENDPOINT,
        "embedding_endpoint_name": embedding_endpoint_name
        }
    else:
      base_parameters = dict()
    task_dict = {
        "existing_cluster_id": cluster_id,
        "notebook_task": NotebookTask(
            base_parameters=base_parameters,
            notebook_path=f"{current_directory}/{task}",
            source=Source("WORKSPACE")
        ),
        "task_key": task
    }
    if i > 0:
        task_dict["depends_on"] = [TaskDependency(tasks_list[i-1])]
    tasks.append(Task(**task_dict))

if job_id:
    print(f"Job already exists and you can view the job at {databricks_url}/#job/{job_id}\n")
    run = w.jobs.run_now(job_id, job_parameters=base_parameters)
    print(f"Running new job run and can view at {databricks_url}/jobs/{job_id}/runs/{run.response.run_id}")
else:
    print("Creating new job....")
    j = w.jobs.create(
        name = job_name,
        tasks = tasks
    )
    run = w.jobs.run_now(j.job_id, job_parameters=base_parameters)

    print(f"View the job at {databricks_url}/#job/{j.job_id}\n")
    print(f"Running new job run and can view at {databricks_url}/jobs/{j.job_id}/runs/{run.response.run_id}")
