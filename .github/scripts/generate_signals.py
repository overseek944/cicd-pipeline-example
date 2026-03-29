"""Generate signals.json from the latest LangSmith experiments."""

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv(override=True)

from langsmith import Client

DATASET_NAME = "text2sql-agent"


def get_latest_experiment_stats(client: Client, prefix: str):
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    except Exception as e:
        print(f"Error reading dataset: {e}", file=sys.stderr)
        return None

    experiments = list(client.list_projects(reference_dataset_id=dataset.id))
    prefix_experiments = [e for e in experiments if e.name.startswith(prefix)]
    
    if not prefix_experiments:
        print(f"No experiments found with prefix {prefix}", file=sys.stderr)
        return None

    prefix_experiments.sort(key=lambda x: x.start_time, reverse=True)
    latest_exp = prefix_experiments[0]
    print(f"Latest {prefix} experiment: {latest_exp.name}", file=sys.stderr)

    # First try getting pre-aggregated stats
    stats = getattr(latest_exp, "feedback_stats", None)
    if stats:
        return {k: v.get("avg", v) if isinstance(v, dict) else v for k, v in stats.items()}

    # If missing, aggregate manually from individual runs
    print("Pre-aggregated feedback_stats missing or empty, calculating from runs...", file=sys.stderr)
    runs = list(client.list_runs(project_name=latest_exp.name, is_root=True))
    if not runs:
        return {}

    run_ids = [r.id for r in runs]
    feedbacks = list(client.list_feedback(run_ids=run_ids))
    
    metric_values = {}
    for f in feedbacks:
        if f.score is not None:
            if f.key not in metric_values:
                metric_values[f.key] = []
            metric_values[f.key].append(f.score)
            
    averages = {}
    for k, v in metric_values.items():
        if v:
            averages[k] = sum(v) / len(v)
            
    return averages


def main():
    client = Client()
    signals = []

    # 1. Fetch SQL Evaluation Stats
    sql_stats = get_latest_experiment_stats(client, "text2sql-agent-sql")
    if sql_stats:
        for metric, avg in sql_stats.items():
            if avg is not None:
                signals.append({
                    "system": "text2sql",
                    "component": "sql_eval",
                    "metric": metric,
                    "value": float(avg)
                })

    # 2. Fetch E2E Agent Evaluation Stats
    e2e_stats = get_latest_experiment_stats(client, "text2sql-agent-e2e")
    if e2e_stats:
        for metric, avg in e2e_stats.items():
            if avg is not None:
                signals.append({
                    "system": "text2sql",
                    "component": "e2e_eval",
                    "metric": metric,
                    "value": float(avg)
                })

    if not signals:
        print("No signals found from LangSmith experiments.", file=sys.stderr)
        sys.exit(1)

    output = {
        "name": "langsmith-evals",
        "version": "1.0.0",
        "signals": signals
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
