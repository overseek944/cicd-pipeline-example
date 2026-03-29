"""Fetch the latest experiment results from LangSmith."""

from dotenv import load_dotenv

load_dotenv(override=True)

from langsmith import Client

client = Client()

DATASET_NAME = "text2sql-agent"


def get_latest_results():
    dataset = client.read_dataset(dataset_name=DATASET_NAME)
    experiments = list(client.list_projects(reference_dataset_id=dataset.id))

    if not experiments:
        print("No experiments found.")
        return

    experiments.sort(key=lambda x: x.start_time, reverse=True)

    for exp in experiments[:5]:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp.name}")
        print(f"Created:    {exp.start_time}")

        # Get feedback stats (aggregated scores)
        if hasattr(exp, "feedback_stats") and exp.feedback_stats:
            print("Scores:")
            for key, stats in exp.feedback_stats.items():
                avg = stats.get("avg", "N/A") if isinstance(stats, dict) else "N/A"
                if isinstance(avg, float):
                    print(f"  {key}: {avg:.3f}")
                else:
                    print(f"  {key}: {avg}")
        else:
            print("Scores: (checking individual runs...)")

        # Get individual run results
        print("\nRun details:")
        runs = list(client.list_runs(project_name=exp.name, is_root=True))
        for i, run in enumerate(runs, 1):
            question = run.inputs.get("question", "N/A") if run.inputs else "N/A"
            output = ""
            if run.outputs:
                output = run.outputs.get("response", run.outputs.get("sql", str(run.outputs)))

            status = "✓" if run.status == "success" else "✗"

            if len(str(output)) > 80:
                output = str(output)[:80] + "..."

            print(f"  {status} [{i}] Q: {question}")
            print(f"       A: {output}")

            # Show feedback for this run
            feedbacks = list(client.list_feedback(run_ids=[run.id]))
            if feedbacks:
                scores = {f.key: f.score for f in feedbacks if f.score is not None}
                if scores:
                    print(f"       Scores: {scores}")


if __name__ == "__main__":
    get_latest_results()
