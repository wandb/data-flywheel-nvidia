# Getting Started with Data Flywheel Blueprint

Learn how to set up and deploy the Data Flywheel Blueprint using the steps in this guide.

This quickstart provides an initial [AIVA dataset](../data/aiva-final.jsonl), and [notebook](../notebooks/monitor_job.ipynb) to help you get started working with the services.

## Prerequisites

### Hardware Requirements

- **Minimum GPU requirements**: 
  - Self-hosted LLM Judge model: 6x (NVIDIA H100 GPUs or NVIDIA A100 GPUs)
  - Remote LLM Judge model: 2x (NVIDIA H100 GPUs or NVIDIA A100 GPUs)
- **Cluster**: A single-node NVIDIA GPU cluster on a Linux host with cluster-admin level permissions
- **Disk space**: At least 200 GB of free disk space

### Software and Access Requirements

Before you begin, make sure you have:

- [Docker and Docker Compose](https://docs.docker.com/desktop) installed
- [Access to NVIDIA NGC](https://catalog.ngc.nvidia.com/) (for API key)

### Obtain an NGC API Key and Log In

You must [generate a personal API key](https://org.ngc.nvidia.com/setup/api-keys) with the `NGC catalog` and `Public API Endpoints` services selected. This enables you to:

- Complete deployment of NMP (NeMo Microservices Platform)
- Access NIM services
- Access models hosted in the NVIDIA API Catalog
- Download models on-premises

For detailed steps, see the official [NGC Private Registry User Guide](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#generating-personal-api-key).


### Install and Configure Git LFS

You must have Git LFS installed and configured to download the dataset files.

1. Download and install Git LFS by following the [installation instructions](https://git-lfs.com/).

2. Initialize Git LFS in your environment.

   ```bash
   git lfs install
   ```

3. Pull the dataset into the current repo.

   ```bash
   git-lfs pull
   ```

---

## Set Up the Data Flywheel Blueprint

### 1. Login to NGC via NVCF

Authenticate with NGC using `nvcf login`. For detailed instructions, see the [NGC Private Registry User Guide on Accessing NGC Registry](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#accessing-ngc-registry).

### 2. Deploy NMP

To deploy NMP, follow the [NeMo Microservices Platform Prerequisites](https://docs.nvidia.com/nemo/microservices/latest/get-started/platform-prereq.html#beginner-tutorial-prerequisites) beginner tutorial. These instructions launch NMP using a local Minikube clsuter. You have two options:

- [Installing using deployment scripts](https://docs.nvidia.com/nemo/microservices/latest/get-started/platform-prereq.html#nemo-ms-get-started-prerequisites-using-deployment-scripts)
- [Installing manually](https://docs.nvidia.com/nemo/microservices/latest/get-started/platform-prereq.html#installing-manually)

> **Important**
> The model deployment (spinning up/down of models in the given namespace) is handled automatically by the Data Flywheel Blueprint and does not require manual intervention. The blueprint manages all aspects of model lifecycle within the configured namespace.

### 3. Configure Data Flywheel

1. Set up the required environment variables:

    Create an NGC API key following the instructions at [Generating NGC API Keys](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#generating-api-key)

   ```bash
   export NGC_API_KEY="<your-ngc-api-key>"
   ```

   For a complete list of environment variables and their descriptions, see the [Environment Variables section](03-configuration.md#environment-variables) in the Configuration Guide.

2. Clone the repository:

   ```bash
   git clone https://gitlab-master.nvidia.com/aire/microservices/data-flywheel-blueprint
   cd data-flywheel-blueprint
   git checkout main
   ```

3. Review and modify the [configuration file](../config/config.yaml) according to your requirements.

   **About the configuration file**

   The `config.yaml` file controls which models (NIMs) are deployed and how the system runs. The main sections are:

   - `nmp_config`: URLs and namespace for your NMP deployment.
   - `nims`: List of models to deploy. Each entry lets you set the model name, context length, GPU count, and other options. Uncomment or add entries to test different models.
   - `data_split_config`: How your data is split for training, validation, and evaluation.
   - `icl_config`: Settings for in-context learning (ICL) examples.
   - `training_config` and `lora_config`: Training and fine-tuning parameters.
   - `logging_config`: Settings got logging . You can configure the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is "INFO".

   **Example: Adding a new NIM**

   ```yaml
   nims:
     - model_name: "meta/llama-3.2-1b-instruct"
       context_length: 32768
       gpus: 1
       pvc_size: 25Gi
       tag: "1.8.3"
       customization_enabled: true
     - model_name: "meta/llama-3.1-8b-instruct"
       context_length: 32768
       gpus: 1
       pvc_size: 25Gi
       tag: "1.8.3"
   ```

   For more details, see the comments in the config file.

### 4. Start Services

You have several options to start the services:

1. **[Recommended]** Using the [launch script](../scripts/run.sh):

   ```bash
   ./scripts/run.sh
   ```


1. Using the [development script](../scripts/run-dev.sh):

   This script runs additional services for observability:

   - `flower`: A web UI for monitoring Celery tasks and workers
   - `kibana`: A visualization dashboard for exploring data stored in Elasticsearch

   ```bash
   ./scripts/run-dev.sh
   ```

1. Using Docker Compose directly:

   ```bash
   docker compose -f ./deploy/docker-compose.yaml up --build
   ```

### 4. Load Data

There are two ways to feed data to the Flywheel:

1. **Manually**: For demo or short-lived environments using the provided `load_test_data.py` script.
1. **Automatically**: For production environments where you deploy the Blueprint to run continuuously, via a [continuous log exportation flow](./01-architecture.md#how-production-logs-flow-into-the-flywheel).

Use the provided script and demo datasets to quickly experience the value of the flywheel service.

#### Demo Data

Load test data using the provided scripts:

##### AIVA Dataset

```bash
uv run python src/scripts/load_test_data.py \
  --file aiva-final.jsonl
```

#### Custom Data

To submit your own custom dataset, provide the loader with a file in [JSON Lines (JSONL)](https://jsonlines.org/) format. The JSONL file should contain one JSON object per line with the following structure:

#### Example Entry

```json
{"messages": [
  {"role": "user", "content": "Describe your issue here."},
  {"role": "assistant", "content": "Assistant's response goes here."}
]}
```

#### Example with Tool Calling

```json
{
  "request": {
    "model": "meta/llama-3.1-70b-instruct", 
    "messages": [
      {"role": "system", "content": "You are a chatbot that helps with purchase history."},
      {"role": "user", "content": "Has my bill been processed?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "structured_rag",
          "parameters": {
            "properties": {
              "query": {"type": "string"},
              "user_id": {"type": "string"}
            },
            "required": ["query", "user_id"]
          }
        }
      }
    ]
  },
  "response": {
    "choices": [{
      "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
          "type": "function",
          "function": {
            "name": "structured_rag",
            "arguments": {"query": "account bill processed", "user_id": "4165"}
          }
        }]
      }
    }]
  },
  "workload_id": "aiva_1",
  "client_id": "dev",
  "timestamp": 1746138417
}
```

Each line in your dataset file should follow this structure, which is compatible with the OpenAI API request and response format.

> **Note**
>
> If `workload_id` and `client_id` are not provided in the dataset entries, they can be specified when running a job.

---

## Run a Job

Now that you have the Data Flywheel running and loaded with data, you can start running jobs.

### Using curl

1. Start a new job:

   ```bash
   curl -X POST http://localhost:8000/api/jobs \
   -H "Content-Type: application/json" \
   -d '{"workload_id": "aiva_1", "client_id": "dev"}'
   ```

   For AIVA dataset:

   ```bash
   curl -X POST http://localhost:8000/api/jobs \
   -H "Content-Type: application/json" \
   -d '{"workload_id": "aiva_1", "client_id": "dev"}'
   ```

2. Check job status and results:

   ```bash
   curl -X GET http://localhost:8000/api/jobs/:job-id -H "Content-Type: application/json"
   ```

   #### Job Response Schema

   When querying a job, you'll receive a JSON response with the following structure:

   ```json
   {
     "id": "65f8a1b2c3d4e5f6a7b8c9d0",          // Unique job identifier
     "workload_id": "aiva_1",               // Workload being processed
     "client_id": "dev",                // Client identifier
     "status": "running",                        // Current job status
     "started_at": "2024-03-15T14:30:00Z",      // Job start timestamp
     "finished_at": "2024-03-15T15:30:00Z",      // Job completion timestamp (if finished)
     "num_records": 1000,                        // Number of processed records
     "llm_judge": { ... },                       // LLM Judge model status
     "nims": [ ... ],                           // List of NIMs and their evaluation results
     "datasets": [ ... ]                        // List of datasets used in the job
   }
   ```

### Using Notebooks

**Note**: Make sure all services are running before accessing the notebook interface.

1. Launch Jupyter Lab using uv:

   ```bash
   uv run jupyter lab \
     --allow-root \
     --ip=0.0.0.0 \
     --NotebookApp.token='' \
     --port=8889 \
     --no-browser
   ```

2. Access Jupyter Lab in your browser at `http://<your-host-ip>:8889`
3. Navigate to the `notebooks` directory
4. Open the example notebook for running and monitoring jobs

Follow the instructions in the notebook to interact with the Data Flywheel services.

## Cleanup

### 1. Data Flywheel

When you are done using the services, you can stop them using the stop script:

```bash
./scripts/stop.sh
```

Then, you can clean up using the [clear volumes script](../scripts/clear_all_volumes.sh):

```bash
./scripts/clear_all_volumes.sh
```

This script will clear all service volumes (Elasticsearch, Redis, and MongoDB).

### 2. NMP Cleanup

You can remove NMP when you are done using the platform by following the official [Uninstall NeMo Microservices Helm Chart](https://docs.nvidia.com/nemo/microservices/latest/set-up/deploy-as-platform/uninstall-platform-helm-chart.html) guide.

## Troubleshooting

If you encounter any issues:

1. Check that all environment variables are properly set
   - See the [Environment Variables section](03-configuration.md#environment-variables) for the complete list of required and optional variables
2. Ensure all prerequisites are installed and configured correctly
3. Verify that you have the necessary permissions and access to all required resources

## Additional Resources

- [Data Flywheel Blueprint Repository](https://gitlab-master.nvidia.com/aire/microservices/data-flywheel-blueprint)
