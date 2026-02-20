from areal.utils import logging

from dataset.const import AnswerType, ProblemType

from .bbox import bbox_reward_fn
from .bool import bool_reward_fn
from .critic import critic_reward_fn
from .format import must_have_bbox_reward_fn
from .generalcode import general_code_reward_fn
from .htmlcode import html_reward_fn
from .math import math_reward_fn
from .multiple_choice import multiplechoice_reward_fn
from .number import number_reward_fn
from .ocr import ocr_reward_fn
from .string_matching import string_matching_reward_fn
from .svgcode import svg_reward_fn
from .flex_match import compute_score

logger = logging.getLogger("Reward System")


REWARD_FUNCTION_MAPPING = {
    AnswerType.NUMBER: number_reward_fn,
    AnswerType.MATH_EXPRESSIONS: math_reward_fn,
    AnswerType.HTML_CODE: html_reward_fn,
    AnswerType.SVG_CODE: svg_reward_fn,
    AnswerType.BOOLEAN: bool_reward_fn,
    AnswerType.MULTIPLE_CHOICE: multiplechoice_reward_fn,
    AnswerType.OCRTEXT: ocr_reward_fn,
    AnswerType.GENERAL_CODE: general_code_reward_fn,
    AnswerType.BBOX: bbox_reward_fn,
    AnswerType.CRITIC: critic_reward_fn,
    AnswerType.ANY: string_matching_reward_fn,
}



class RewardSystem:
    def __init__(self):
        pass

    def reward(
        self,
        prompt: str,
        completions: str,
        answer: str | list[str],
        answer_type: AnswerType = None,
        *args,
        **kwargs
    ):
        if isinstance(answer, list) and len(answer) == 1:
            answer = answer[0]
        elif isinstance(answer, list):
            answer = str(answer)

        logger.debug(f"========================\nPrompt: {prompt}\n----------------------\nCompletions: {completions}\n---------------------\nAnswers: {answer}")

        if answer_type not in REWARD_FUNCTION_MAPPING:
            # logger.warning(f"Unknown answer type: {answer_type}. Using string matching reward as default.")
            extra_info = {"question": prompt}
            score_dict = compute_score(data_source="rl", solution_str=completions, ground_truth=answer, extra_info=extra_info)
            acc_score = score_dict["acc_score"]
            format_score = score_dict["format_reward_score"]

            rewards = {
                "acc_reward": acc_score,
                "format_reward": format_score,
                "reward": score_dict["score"]
            }
        else:
            reward_fn = REWARD_FUNCTION_MAPPING.get(answer_type, string_matching_reward_fn)
            scores = reward_fn(completions, answer)

            rewards = {
                "acc_reward": scores,
            }

            if "problem_type" in kwargs and str(kwargs["problem_type"]) == str(ProblemType.SPATIAL_REASONING):
                bbox_reward = must_have_bbox_reward_fn(completions)
                scores = scores + bbox_reward * 0.5
                rewards["bbox_reward"] = bbox_reward
            
            rewards["reward"] = scores

        logger.debug(f"Reward Scores: {rewards}")
        return rewards