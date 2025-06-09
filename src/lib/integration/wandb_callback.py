import wandb
import requests
from datetime import datetime
import pandas as pd

from src.api.db_manager import TaskDBManager
from src.log_utils import setup_logging

logger = setup_logging("data_flywheel.wandb_callback")

# --- Functions replicated from notebooks/job_monitor_helper.py ---
def format_runtime(seconds):
    """Format runtime in seconds to a human-readable string."""
    if seconds is None:
        return "-"
    minutes, seconds = divmod(seconds, 60)
    if minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    return f"{int(seconds)}s"

def get_job_status(job_id, api_base_url="http://0.0.0.0:8000"):
    """Get the current status of a job."""
    url = f"{api_base_url.rstrip('/')}/api/jobs/{job_id}"
    logger.debug(f"Fetching job status from: {url}")
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def create_results_table(job_data):
    """Create a pandas DataFrame from job data's evaluations."""
    rows = []
    for nim in job_data.get("nims", []):
        model_name = nim.get("model_name")
        for eval_item in nim.get("evaluations", []):
            all_scores = eval_item.get("scores", {})
            
            row = {
                "Model": model_name,
                "Eval Type": eval_item.get("eval_type", "").upper(),
                "Percent Done": eval_item.get("progress"),
                "Runtime": format_runtime(eval_item.get("runtime_seconds")),
                "Status": "Completed" if eval_item.get("finished_at") else "Running",
                "Started": datetime.fromisoformat(eval_item["started_at"]).strftime("%H:%M:%S") if eval_item.get("started_at") else "-",
                "Finished": datetime.fromisoformat(eval_item["finished_at"]).strftime("%H:%M:%S") if eval_item.get("finished_at") else "-",
                "ID": eval_item.get("id") # Added ID for easier reference
            }
            
            # Standard scores
            if "function_name" in all_scores:
                row["Function name accuracy"] = all_scores["function_name"]
            if "function_name_and_args_accuracy" in all_scores:
                row["Function name + args accuracy (exact-match)"] = all_scores["function_name_and_args_accuracy"]
            if "tool_calling_correctness" in all_scores:
                row["Function name + args accuracy (LLM-judge)"] = all_scores["tool_calling_correctness"]
            
            # Add any other scores with formatted names
            for score_name, score_value in all_scores.items():
                if score_name not in row: # Avoid overwriting already processed scores
                    formatted_name = score_name.replace("_", " ").title()
                    row[formatted_name] = score_value
            
            rows.append(row)
    
    if not rows:
        return pd.DataFrame(columns=["Model", "Eval Type", "Percent Done", "Runtime", "Status", "Started", "Finished", "ID"])
    
    df = pd.DataFrame(rows)
    return df.sort_values(["Model", "Eval Type"])

def create_customization_table(job_data):
    """Create a pandas DataFrame from job data's customizations."""
    customizations_rows = []
    for nim in job_data.get("nims", []):
        model_name = nim.get("model_name")
        for custom in nim.get("customizations", []):
            customizations_rows.append({
                "Model": model_name,
                "Started": datetime.fromisoformat(custom["started_at"]).strftime("%H:%M:%S") if custom.get("started_at") else "-",
                "Epochs Completed": custom.get("epochs_completed"),
                "Steps Completed": custom.get("steps_completed"),
                "Finished": datetime.fromisoformat(custom["finished_at"]).strftime("%H:%M:%S") if custom.get("finished_at") else "-",
                "Status": "Completed" if custom.get("finished_at") else "Running",
                "Runtime": format_runtime(custom.get("runtime_seconds")),
                "Percent Done": custom.get("progress"),
                "ID": custom.get("id") # Added ID
            })
   
    if not customizations_rows:
        return pd.DataFrame(columns=["Model", "Started", "Epochs Completed", "Steps Completed", "Finished", "Runtime", "Percent Done", "ID"])
    
    df = pd.DataFrame(customizations_rows)
    return df.sort_values(["Model"])
# --- End of replicated functions ---


def make_wandb_progress_callback(manager: TaskDBManager, run, eval_instance, previous_result):
    
    def callback(update_data):
        """
        Update evaluation document in DB, then fetch full job status from API,
        process it using helper logic, and log structured data to W&B.
        """
        logger.debug(f"Callback received update_data for eval_instance.id {eval_instance.id}: {update_data}")
        
        # 1. Update MongoDB with the incremental update_data
        manager.update_evaluation(eval_instance.id, update_data)
        
        try:
            # 2. Fetch the entire job status from the API using the flywheel run ID
            job_data = get_job_status(job_id=previous_result.flywheel_run_id, api_base_url="http://0.0.0.0:8000")
            logger.info(f"Successfully fetched full job data for flywheel_run_id {previous_result.flywheel_run_id}")

            # 3. Process job data using the replicated helper functions
            results_df = create_results_table(job_data)
            customizations_df = create_customization_table(job_data)

            wandb_log_data = {}

            # Log overall job information
            wandb_log_data["job_overall_status"] = job_data.get("status")
            wandb_log_data["job_total_records"] = job_data.get("num_records")
            wandb_log_data["job_last_api_fetch_time"] = datetime.now().isoformat()

            # Log the tables
            if not results_df.empty:
                wandb_log_data["evaluation_summary_table"] = wandb.Table(dataframe=results_df)
            if not customizations_df.empty:
                wandb_log_data["customization_summary_table"] = wandb.Table(dataframe=customizations_df)

            # 4. Log specific metrics for the current eval_instance that triggered this callback
            current_eval_metrics = {}
            for nim_item in job_data.get("nims", []):
                for e_data in nim_item.get("evaluations", []):
                    if e_data.get("id") == eval_instance.id:
                        current_eval_metrics["current_eval_progress"] = e_data.get("progress")
                        current_eval_metrics["current_eval_runtime_seconds"] = e_data.get("runtime_seconds")
                        current_eval_metrics["current_eval_status"] = "Completed" if e_data.get("finished_at") else "Running"
                        for score_name, score_value in e_data.get("scores", {}).items():
                            current_eval_metrics[f"current_eval_score_{score_name}"] = score_value
                        if e_data.get("error"):
                             current_eval_metrics["current_eval_error"] = e_data.get("error")
                        break
                if current_eval_metrics: # Found and processed current eval
                    break
            
            if current_eval_metrics:
                wandb_log_data.update(current_eval_metrics)
            else:
                logger.warning(f"Could not find specific data for eval_instance.id {eval_instance.id} in fetched job_data.")

            # 5. Log to W&B
            if wandb_log_data:
                run.log(wandb_log_data)
                logger.info(f"Logged processed job data to W&B for eval_instance.id {eval_instance.id}. Logged keys: {list(wandb_log_data.keys())}")

        except requests.exceptions.RequestException as e:
            logger.error(f"W&B Callback: Failed to fetch job status for job_id {previous_result.flywheel_run_id}: {e}")
        except Exception as e:
            logger.error(f"W&B Callback: An unexpected error occurred while processing job_id {previous_result.flywheel_run_id}: {e}", exc_info=True)

    return callback
