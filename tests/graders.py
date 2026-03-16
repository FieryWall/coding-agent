from typing import TypeVar
from langsmith import traceable
from coding_agent.llm import LLMBackend, extract_json

T = TypeVar('T')

class BaseGrader:
    def __init__(self, client: LLMBackend):
        self.client = client

    async def grade[T](self, output: str) -> T:
        pass

class JudgeGrader(BaseGrader):

    @traceable(run_type="chain", name="judge_grader")
    async def grade(self, output: str, judgement: str) -> float:
        response = await self.client.acomplete(f"""
            You are a judge.
            Judge how well the model output matches the judgement.
            Score from 0 to 100.
            Return JSON with your reasoning and score.

            Example 1:
            Model output: "The sky is blue."
            Judgement: "Is model is correct?"
            {{"reasoning": "The model is right.", "score": 100}}

            Example 2:
            Model output: "<a></b></a>"
            Judgement: "Is model's output XML valid?"
            {{"reasoning": "The model's output is not valid XML.", "score": 0}}

            Example 3:
            Model output: "Hi."
            Judgement: "How polite is the model's output?"
            {{"reasoning": "The model's output is polite, but might be better.", "score": 34}}

            Model output: {output}
            Judgement: {judgement}

            Return valid JSON with your reasoning and score only. Do not return anything else.
        """)

        data = extract_json(response)
        if not data:
            return 0.0

        raw_score = data.get("score")
        try:
            score = float(raw_score)
        except (ValueError, TypeError):
            return 0.0
        if score > 1.0:
            score = score / 100.0
        return max(0.0, min(1.0, score))


class FactJudgeGrader(BaseGrader):
    @traceable(run_type="chain", name="fact_judge_grader")
    async def grade(self, output: str, question: str) -> bool:
        response = await self.client.acomplete(f"""
            Answer strictly YES or NO based on the following fact check.
            Output to evaluate: {output}
            Question: {question}
            Reply with exactly one word: YES or NO.
        """)
        text = (response or "").strip().upper()
        return "YES" in text[:4] and "NO" not in text[:3]


class PairwiseComparisonGrader(BaseGrader):
    @traceable(run_type="chain", name="pairwise_grader")
    async def grade(self, output_a: str, output_b: str, criterion: str) -> float:
        response = await self.client.acomplete(f"""
            Compare two responses. Criterion: {criterion}
            Response A: {output_a}
            Response B: {output_b}
            Rate how much A is better than B: 0-100 (100 = A much better, 50 = tie, 0 = B much better).
            Return JSON only: {{"score": <0-100>}}
        """)
        data = extract_json(response)
        if not data:
            return 0.5
        try:
            score = float(data.get("score", 50))
        except (ValueError, TypeError):
            return 0.5
        if score > 1.0:
            score = score / 100.0
        return max(0.0, min(1.0, score))