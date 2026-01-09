from pydantic import BaseModel

class EvaluationConfig(BaseModel):
    experiment_name: str
    description: str
    max_concurrency: int
    save_results: bool
    save_results_path: str

    