import os
from datetime import datetime
from dotenv import load_dotenv
from coding_agent.llm import create_llm_client, PRODUCTION_LLM

load_dotenv()
from coding_agent.tools import ToolsRepository
from tests.graders import JudgeGrader, FactJudgeGrader, PairwiseComparisonGrader
import pytest

TEST_LLM = "gpt-4.1-mini"
NANO_LLM = "gpt-4.1-nano"


def pytest_addoption(parser):
    parser.addoption(
        "--model",
        action="store",
        default=None,
        help="Path to local LLM model (GGUF or MLX directory)",
    )


def _get_langsmith_client():
    try:
        from langsmith import Client
        return Client()
    except Exception:
        return None


@pytest.fixture(scope="session")
def langsmith_client():
    return _get_langsmith_client()


@pytest.fixture(scope="session")
def production_llm_client():
    return create_llm_client(model_name=PRODUCTION_LLM)


@pytest.fixture(scope="session")
def nano_llm_client():
    return create_llm_client(model_name=NANO_LLM)


@pytest.fixture
def tools_repo():
    return ToolsRepository()


@pytest.fixture(scope="session")
def judge_grader(request):
    model_path = request.config.getoption("--model")
    test_llm_client = create_llm_client(model_path) if model_path else create_llm_client(model_name=TEST_LLM)
    return JudgeGrader(test_llm_client)


@pytest.fixture(scope="session")
def fact_judge_grader(request):
    model_path = request.config.getoption("--model")
    test_llm_client = create_llm_client(model_path) if model_path else create_llm_client(model_name=TEST_LLM)
    return FactJudgeGrader(test_llm_client)


@pytest.fixture(scope="session")
def pairwise_grader(request):
    model_path = request.config.getoption("--model")
    test_llm_client = create_llm_client(model_path) if model_path else create_llm_client(model_name=TEST_LLM)
    return PairwiseComparisonGrader(test_llm_client)


class EvalResultCollector:
    def __init__(self):
        self.results = []
        self.experiment_name = f"coding-agent-evals-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    def add_result(self, test_name: str, passed: bool, score: float = None, metadata: dict = None):
        self.results.append({
            "test_name": test_name,
            "passed": passed,
            "score": score if score is not None else (1.0 if passed else 0.0),
            "metadata": metadata or {},
        })
    
    def summary(self) -> dict:
        if not self.results:
            return {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}
        passed = sum(1 for r in self.results if r["passed"])
        return {
            "total": len(self.results),
            "passed": passed,
            "failed": len(self.results) - passed,
            "pass_rate": passed / len(self.results),
            "avg_score": sum(r["score"] for r in self.results) / len(self.results),
        }


@pytest.fixture(scope="session")
def eval_collector():
    return EvalResultCollector()


@pytest.fixture(scope="session")
def langsmith_eval_runner():
    from tests.langsmith_evals import LangSmithEvalRunner
    return LangSmithEvalRunner()


def pytest_configure(config):
    config.addinivalue_line("markers", "eval: mark test as an eval test for LangSmith tracking")