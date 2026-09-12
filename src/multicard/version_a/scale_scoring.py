"""Pure scoring helpers. No network or model execution.

BEAM rubric prompt reproduced from the MIT-licensed official implementation:
https://github.com/mohammadtavakoli78/BEAM/blob/b2da22eac88bb0874c64665f13457eb99835774a/src/prompts.py
MAB normalization follows official utils/eval_other_utils.py at
fe1735de8cf8b9908e1e3d3b5612afc815698062.
"""
from __future__ import annotations
import json
import math
import re
import string


def normalize_answer(text: str) -> str:
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def substring_exact_match(prediction: str, references: list[str]) -> float:
    """Official MAB maximum normalized substring accuracy, not token F1."""
    normalized = normalize_answer(prediction)
    return float(any(normalize_answer(reference) in normalized for reference in references))


def beam_rubric_prompts(question: dict, prediction: str) -> list[str]:
    """Exact official prompt: one model request per rubric item, not per answer."""
    rubric = question.get("rubric")
    if not isinstance(rubric, list) or not rubric:
        raise ValueError("BEAM question needs nonempty rubric")
    # Official prompt mentions QUESTION but does not interpolate the question.
    # Preserve this behavior for protocol fidelity and disclose it in reports.
    return [BEAM_JUDGE_PROMPT.replace("<rubric_item>", item).replace("<llm_response>", prediction)
            for item in rubric]


def parse_beam_rubric(response: str) -> dict:
    text = response.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("BEAM judge must return an object")
    score = parsed.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or score not in (0, 0.5, 1):
        raise ValueError("BEAM rubric score must be 0, 0.5, or 1")
    return {"score": float(score), "reason": str(parsed.get("reason", ""))}


def beam_rubric_mean(judgements: list[dict], expected_items: int) -> float:
    if expected_items <= 0 or len(judgements) != expected_items:
        raise ValueError("All rubric items must be judged; partial scores cannot be reported")
    scores = [parse_beam_rubric(json.dumps(item))["score"] for item in judgements]
    return sum(scores) / expected_items


def beam_equivalence_messages(candidate: str, reference: str) -> list[dict]:
    """Official event-alignment prompt; iterate candidate-first, first unused match."""
    return [
        {"role": "system", "content": """
            You are a binary classifier.
            If the TWO snippets describe the SAME event/fact, reply **YES**
            Otherwise reply **NO**. No extra words.
            DO NOT provide any exaplanation.
        """},
        {"role": "user", "content": f"""First snippet: {reference} \n
                       Second snippet: {candidate}
                    """},
    ]


def beam_event_ordering(reference: list[str], aligned_prediction: list[str]) -> dict:
    """Official event formula after external one-to-one semantic alignment.

    Keep nonmatched prediction strings; do not drop them or pre-sort events.
    Undefined tau is returned as None instead of reporting invalid JSON NaN.
    """
    from scipy.stats import kendalltau
    true_positive = len(set(reference) & set(aligned_prediction))
    false_positive = sum(item not in reference for item in aligned_prediction)
    false_negative = sum(item not in aligned_prediction for item in reference)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = list(dict.fromkeys(reference + aligned_prediction))
    tie = len(union) + 1
    def ranks(items):
        positions = {item: i + 1 for i, item in enumerate(items)}
        return [positions.get(item, tie) for item in union]
    tau = kendalltau(ranks(reference), ranks(aligned_prediction), variant="b", method="auto").statistic
    normalized = (float(tau) + 1) / 2 if math.isfinite(tau) else None
    return {"precision": precision, "recall": recall, "f1": f1, "tau_norm": normalized,
            "final_score": normalized * f1 if normalized is not None else None}


def evaluate_beam(q: dict, pred: str, judge) -> dict:
    """Execute official rubric-item judgement and optional event alignment.

    Model identity and every call's usage remain in the caller's metered client.
    The client's single-user-message interface joins the official alignment
    system/user text; this transport difference must remain disclosed.
    No binary pass threshold is defined by this evaluator.
    """
    calls, judgements = [], []
    for prompt in beam_rubric_prompts(q, pred):
        response = judge.generate(prompt, 512, role="judge")
        calls.append(response)
        judgements.append(parse_beam_rubric(response["text"]))
    result = {"score": beam_rubric_mean(judgements, len(q["rubric"])),
              "supplementary": {}, "calls": calls}
    if q["type"] != "event_ordering":
        return result
    reference = q["rubric"]
    used = set()
    aligned = []
    # The official source splits the answer by newline, including blank lines.
    # It greedily accepts the first unused semantically equivalent reference.
    for candidate in pred.split("\n"):
        matched = None
        for index, item in enumerate(reference):
            if index in used:
                continue
            messages = beam_equivalence_messages(candidate, item)
            prompt = "\n\n".join(message["content"] for message in messages)
            response = judge.generate(prompt, 64, role="judge")
            calls.append(response)
            verdict = response["text"].strip().lower().rstrip(".")
            if verdict not in ("yes", "no"):
                raise ValueError("Unparseable BEAM event equivalence judgement")
            if verdict == "yes":
                matched = index
                break
        if matched is None:
            aligned.append(candidate)
        else:
            used.add(matched)
            aligned.append(reference[matched])
    result["supplementary"] = beam_event_ordering(reference, aligned)
    result["supplementary"]["aligned_prediction"] = aligned
    return result

BEAM_JUDGE_PROMPT = '\nYou are an expert evaluator tasked with judging whether the LLM\'s response demonstrates compliance with the specified RUBRIC CRITERION.\n\n## EVALUATION INPUTS\n- RUBRIC CRITERION (what to check): <rubric_item>\n- RESPONSE TO EVALUATE: <llm_response>\n\n## EVALUATION RUBRIC:\nThe rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate. \n\n**IMPORTANT**: Pay careful attention to whether the rubric specifies:\n- **Positive requirements** (things the response SHOULD include/do)\n- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")\n\n## RESPONSIVENESS REQUIREMENT\nA compliant response must be **on-topic** and attempt to answer it.\n- If the response does not address the QUESTION, score **0.0** and stop.\n- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.\n\n## SEMANTIC TOLERANCE RULES:\nJudge by meaning, not exact wording.\n- Accept **paraphrases** and **synonyms** that preserve intent.\n- **Case/punctuation/whitespace** differences must be ignored.\n- **Numbers/currencies/dates** may appear in equivalent forms (e.g., “$68,000”, “68k”, “68,000 USD”, or “sixty-eight thousand dollars”). Treat them as equal when numerically equivalent.\n- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.\n\n## STYLE NEUTRALITY (prevents style contamination):\nIgnore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., “itemized list”, “no citations”, “one sentence”).\n- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.\n- Only evaluate format when the rubric **explicitly** mandates it.\n\n## SCORING SCALE:\n- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.\n  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).\n  - Negative: prohibited element **absent** AND response is **responsive**.\n  \n- **0.5 (Partial Compliance)**: Partially complies.\n  - Positive: element present but minor inaccuracies/incomplete execution.\n  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.\n  \n- **0.0 (No Compliance)**: Fails to comply.\n  - Positive: required element missing or incorrect.\n  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.\n\n## EVALUATION INSTRUCTIONS:\n1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).\n\n2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:\n   - **All elements** must be present for full compliance (1.0)\n   - **Some elements** present indicates partial compliance (0.5)\n   - **No elements** present indicates no compliance (0.0)\n   \n3. **Check Compliance**: \n   - For positive requirements: Look for the presence and quality of the required element\n   - For negative constraints: Look for the absence of the prohibited element\n\n4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.\n\n5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.\n\n## OUTPUT FORMAT:\nReturn your evaluation in JSON format with two fields:\n\n{\n   "score": [your score: 1.0, 0.5, or 0.0],\n   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"\n}\n\nNOTE: ONLY output the json object, without any explanation before or after that\n'
