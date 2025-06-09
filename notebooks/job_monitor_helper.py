import matplotlib.pyplot as plt
import pandas as pd
import time
import requests
from datetime import datetime
from IPython.display import clear_output
import json

def format_runtime(seconds):
    """Format runtime in seconds to a human-readable string."""
    if seconds is None:
        return "-"
    minutes, seconds = divmod(seconds, 60)
    if minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    return f"{int(seconds)}s"

def create_results_table(job_data):
    """Create a pandas DataFrame from job data."""
    print("\n=== DEBUG: Job Data Structure ===")
    print(f"Job Status: {job_data.get('status')}")
    print(f"Total Records: {job_data.get('num_records')}")
    print(f"Number of NIMs: {len(job_data.get('nims', []))}")
    
    rows = []
    for nim_idx, nim in enumerate(job_data.get("nims", [])):
        print(f"\n--- NIM {nim_idx + 1} ---")
        print(f"Model Name: {nim.get('model_name')}")
        print(f"Number of Evaluations: {len(nim.get('evaluations', []))}")
        
        for eval_idx, eval in enumerate(nim.get("evaluations", [])):
            print(f"\n  Evaluation {eval_idx + 1}:")
            print(f"  Eval Type: {eval.get('eval_type')}")
            print(f"  Progress: {eval.get('progress')}")
            print(f"  Runtime: {eval.get('runtime_seconds')}")
            print(f"  Started At: {eval.get('started_at')}")
            print(f"  Finished At: {eval.get('finished_at')}")
            print(f"  Scores: {json.dumps(eval.get('scores', {}), indent=2)}")
            
            all_scores = eval.get("scores", {})
            
            row = {
                "Model": nim.get("model_name"),
                "Eval Type": eval.get("eval_type", "").upper(),
                "Percent Done": eval.get("progress"),
                "Runtime": format_runtime(eval.get("runtime_seconds")),
                "Status": "Completed" if eval.get("finished_at") else "Running",
                "Started": datetime.fromisoformat(eval["started_at"]).strftime("%H:%M:%S") if eval.get("started_at") else "-",
                "Finished": datetime.fromisoformat(eval["finished_at"]).strftime("%H:%M:%S") if eval.get("finished_at") else "-"
            }
            
            if "function_name" in all_scores:
                row["Function name accuracy"] = all_scores["function_name"]
            
            if "function_name_and_args_accuracy" in all_scores:
                row["Function name + args accuracy (exact-match)"] = all_scores["function_name_and_args_accuracy"]
                
            if "tool_calling_correctness" in all_scores:
                row["Function name + args accuracy (LLM-judge)"] = all_scores["tool_calling_correctness"]
            
            # Add any other scores with formatted names
            for score_name, score_value in all_scores.items():
                if score_name not in ["function_name", "tool_calling_correctness", "similarity", "function_name_and_args_accuracy"]:
                    formatted_name = score_name.replace("_", " ").title()
                    row[formatted_name] = score_value
            
            rows.append(row)
    
    if not rows:
        print("\nNo evaluation data found!")
        return pd.DataFrame(columns=["Model", "Eval Type", "Function Name Accuracy", "Tool Calling Correctness (LLM-Judge)", "Similarity (LLM-Judge)", "Percent Done", "Runtime", "Status", "Started", "Finished"])
    
    df = pd.DataFrame(rows)
    return df.sort_values(["Model", "Eval Type"])

def create_customization_table(job_data):
    """Create a pandas DataFrame from customization data."""
    print("\n=== DEBUG: Customization Data ===")
    customizations = []
    for nim_idx, nim in enumerate(job_data.get("nims", [])):
        print(f"\n--- NIM {nim_idx + 1} Customizations ---")
        print(f"Model Name: {nim.get('model_name')}")
        print(f"Number of Customizations: {len(nim.get('customizations', []))}")
        
        for custom_idx, custom in enumerate(nim.get("customizations", [])):
            print(f"\n  Customization {custom_idx + 1}:")
            print(f"  Started At: {custom.get('started_at')}")
            print(f"  Epochs Completed: {custom.get('epochs_completed')}")
            print(f"  Steps Completed: {custom.get('steps_completed')}")
            print(f"  Finished At: {custom.get('finished_at')}")
            print(f"  Runtime: {custom.get('runtime_seconds')}")
            print(f"  Progress: {custom.get('progress')}")
            
            customizations.append({
                "Model": nim.get("model_name"),
                "Started": datetime.fromisoformat(custom["started_at"]).strftime("%H:%M:%S") if custom.get("started_at") else "-",
                "Epochs Completed": custom.get("epochs_completed"),
                "Steps Completed": custom.get("steps_completed"),
                "Finished": datetime.fromisoformat(custom["finished_at"]).strftime("%H:%M:%S") if custom.get("finished_at") else "-",
                "Status": "Completed" if custom.get("finished_at") else "Running",
                "Runtime": format_runtime(custom.get("runtime_seconds")),
                "Percent Done": custom.get("progress"),
            })
   
    if not customizations:
        print("\nNo customization data found!")
        return pd.DataFrame(columns=["Model", "Started", "Epochs Completed", "Steps Completed", "Finished", "Runtime", "Percent Done"])
    
    customizations = pd.DataFrame(customizations)
    return customizations.sort_values(["Model"])

def get_job_status(api_base_url, job_id):
    """Get the current status of a job."""
    print(f"\n=== DEBUG: Fetching Job Status ===")
    print(f"API URL: {api_base_url}/api/jobs/{job_id}")
    response = requests.get(f"{api_base_url}/api/jobs/{job_id}")
    response.raise_for_status()
    job_data = response.json()
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    return job_data

def monitor_job(api_base_url, job_id, poll_interval):
    """Monitor a job and display its progress in a table."""
    print(f"Monitoring job {job_id}...")
    print("Press Ctrl+C to stop monitoring")
    
    while True:
        try:
            clear_output(wait=True)
            job_data = get_job_status(api_base_url, job_id)
            results_df = create_results_table(job_data)
            customizations_df = create_customization_table(job_data)
            clear_output(wait=True)
            print(f"Job Status: {job_data['status']}")
            print(f"Total Records: {job_data['num_records']}")
            print(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")
            print("\nResults:")
            display(results_df)
            print("\nCustomizations:")
            display(customizations_df)
            print("\nRaw Job Data:")
            print(json.dumps(job_data, indent=2))

            # Plot 1: Evaluation Scores
            if not results_df.empty:
                metrics = [
                    "Function name accuracy",
                    "Function name + args accuracy (exact-match)",
                    "Function name + args accuracy (LLM-judge)"
                ]
            
                models = results_df["Model"].unique()
            
                for model in models:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    model_df = results_df[results_df["Model"] == model]
                    if not all(metric in model_df.columns for metric in metrics):
                        continue  # skip this model for now                 
                    plot_df = model_df.set_index("Eval Type")[metrics].T
            
                    # Plot bar chart for this model
                    plot_df.plot(kind="bar", ax=ax)
                    ax.set_title(f"Evaluation results for {model}", fontsize=12)
                    ax.set_ylabel("Score")
                    ax.set_ylim(0, 1)
                    ax.legend(title="Eval Type")
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.xticks(rotation=30)
                    plt.tight_layout()
                    plt.show()
            else:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.text(0.5, 0.5, "No Evaluation Data", ha='center', va='center')
                ax.set_axis_off()
                plt.tight_layout()
                plt.show()

            plt.tight_layout()
            plt.show()                        
            time.sleep(poll_interval)

            # Check if job is completed or failed
            if job_data['status'] in ['completed', 'failed']:
                if job_data['status'] == 'failed':
                    print("Job failed - check error details above")
                    if job_data.get('error'):
                        print(f"Error: {job_data['error']}")
                else:
                    print("Job completed successfully!")
                break

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            break