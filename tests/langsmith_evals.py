import asyncio
from typing import Callable, Any, Optional
from dataclasses import dataclass, field
from langsmith import Client, traceable
from langsmith.schemas import Example, Run


@dataclass
class EvalCase:
    name: str
    inputs: dict
    expected: Any = None
    metadata: dict = field(default_factory=dict)


@dataclass 
class EvalResult:
    name: str
    passed: bool
    score: float
    output: Any = None
    error: str = None
    metadata: dict = field(default_factory=dict)


class LangSmithEvalRunner:
    def __init__(self, project_name: str = "coding-agent"):
        self.client = Client()
        self.project_name = project_name
        self.results: list[EvalResult] = []
    
    @traceable(run_type="chain", name="eval_run")
    async def run_eval(
        self,
        case: EvalCase,
        target_fn: Callable,
        evaluator_fn: Callable[[Any, Any], tuple[bool, float]],
    ) -> EvalResult:
        try:
            if asyncio.iscoroutinefunction(target_fn):
                output = await target_fn(**case.inputs)
            else:
                output = target_fn(**case.inputs)
            
            passed, score = evaluator_fn(output, case.expected)
            
            result = EvalResult(
                name=case.name,
                passed=passed,
                score=score,
                output=output,
                metadata=case.metadata,
            )
        except Exception as e:
            result = EvalResult(
                name=case.name,
                passed=False,
                score=0.0,
                error=str(e),
                metadata=case.metadata,
            )
        
        self.results.append(result)
        return result
    
    async def run_all(
        self,
        cases: list[EvalCase],
        target_fn: Callable,
        evaluator_fn: Callable,
    ) -> list[EvalResult]:
        results = []
        for case in cases:
            result = await self.run_eval(case, target_fn, evaluator_fn)
            results.append(result)
        return results
    
    def summary(self) -> dict:
        if not self.results:
            return {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}
        
        passed = sum(1 for r in self.results if r.passed)
        scores = [r.score for r in self.results]
        
        return {
            "total": len(self.results),
            "passed": passed,
            "failed": len(self.results) - passed,
            "pass_rate": passed / len(self.results),
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
        }
    
    def create_dataset(self, dataset_name: str, cases: list[EvalCase]) -> str:
        try:
            dataset = self.client.create_dataset(
                dataset_name=dataset_name,
                description=f"Eval dataset for {self.project_name}",
            )
            
            for case in cases:
                self.client.create_example(
                    inputs=case.inputs,
                    outputs={"expected": case.expected} if case.expected else None,
                    metadata=case.metadata,
                    dataset_id=dataset.id,
                )
            
            return dataset.id
        except Exception as e:
            print(f"Failed to create dataset: {e}")
            return None


def not_none_evaluator(output: Any, expected: Any) -> tuple[bool, float]:
    passed = output is not None
    return passed, 1.0 if passed else 0.0


def equals_evaluator(output: Any, expected: Any) -> tuple[bool, float]:
    passed = output == expected
    return passed, 1.0 if passed else 0.0


def contains_evaluator(output: Any, expected: str) -> tuple[bool, float]:
    if output is None:
        return False, 0.0
    output_str = str(output)
    passed = expected in output_str
    return passed, 1.0 if passed else 0.0


def min_length_evaluator(min_len: int):
    def evaluator(output: Any, expected: Any) -> tuple[bool, float]:
        if output is None:
            return False, 0.0
        length = len(output) if hasattr(output, '__len__') else 0
        passed = length >= min_len
        score = min(1.0, length / min_len) if min_len > 0 else 1.0
        return passed, score
    return evaluator
