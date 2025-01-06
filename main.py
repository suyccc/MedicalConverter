import base64
import io
from textwrap import dedent
import json
import gradio as gr
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import trange, tqdm
import os
import click
from prompts import (
    concept_generation_system_prompt,
    data_processing_generation_system_prompt,
    evaluator_system_prompt,
    fusion_generation_system_prompt,
    question_bias_generation_system_prompt,
    reasoning_generation_system_prompt,
    refine_system_prompt_concept,
    refine_system_prompt_data,
    refine_system_prompt_question_bias,
    refine_system_prompt_reason,
    refine_system_prompt_visual,
    refiner_system_prompt,
    review_system_prompt,
    visual_interpretation_generation_system_prompt,
)
import random
random.seed(1)


class Distractor(BaseModel):
    text: str
    reason: str


class Distractors(BaseModel):
    distractors: list[Distractor]


class Comment(BaseModel):
    option: str
    comment: str


class CommentFormat(BaseModel):
    comments: list[Comment]


class Judgement(BaseModel):
    reasoning: str
    correctness: int
    improvement: str


class Question(BaseModel):
    reasoning: str
    distractors: list[str]


def base64_to_image(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return image


def get_reply(client, system_prompt, user_prompt, image_base64, output_format):
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": dedent(system_prompt)},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": dedent(user_prompt)},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                    ],
                },
            ],
            response_format=output_format,
            # temperature=0,  # Set to 0 for deterministic responses
        )
        parsed_output = completion.choices[0].message.parsed.dict()
        return parsed_output
    except Exception as e:
        
        print("Input: ", user_prompt)
        print("Error: ", e)
        return None


def convert_to_multi_choice(client, question, answer, image_base64, reviewer):
    user_prompt_generate = f"""
    Question: {question}
    Correct Answer: {answer}
    """
    user_prompt_review = """
        Question: {question}
        Correct Answer: {answer}
        Distractions and Reasonings: {distractors}
    """
    user_prompt_refine = """
        Question: {question}
        Correct Answer: {answer}
        Distractions and Reviewer Comments: {reviews}
    """
    error_types = ['concept', 'reasoning', 'visual_interpretation', 'data_processing', 'question_bias']
    generation_system_prompts = {
        "concept": concept_generation_system_prompt,
        "reasoning": reasoning_generation_system_prompt,
        "visual_interpretation": visual_interpretation_generation_system_prompt,
        "data_processing": data_processing_generation_system_prompt,
        "question_bias": question_bias_generation_system_prompt,
        "fusion": fusion_generation_system_prompt,
    }
    refine_system_prompts = {
        "concept": refine_system_prompt_concept,
        "reasoning": refine_system_prompt_reason,
        "visual_interpretation": refine_system_prompt_visual,
        "data_processing": refine_system_prompt_data,
        "question_bias": refine_system_prompt_question_bias,
    }
    final_distractors = {}
    for error_type in error_types:
        ############## Step 1: Generate Distractors #############
        distractors = get_reply(
            client, generation_system_prompts[error_type], user_prompt_generate, image_base64, Distractors
        )["distractors"]
        if distractors is None:
            return None
        if reviewer:
            ############## Step 2: Reivew Distractors #############
            reviews = get_reply(
                client,
                review_system_prompt.format(type=error_type),
                user_prompt_review.format(
                    question=question, answer=answer, distractors=distractors
                ),
                image_base64,
                CommentFormat,
            )["comments"]
            if reviews is None:
                return None
            ############## Step 3: Refine Distractors #############
            refined_distractors = get_reply(
                client,
                refine_system_prompts[error_type],
                user_prompt_refine.format(
                    question=question, answer=answer, reviews=reviews
                ),
                image_base64,
                Distractors,
            )["distractors"]
            if refined_distractors is None:
                return None
            distractors = refined_distractors
        final_distractors[error_type] = distractors
        

    distractors = (final_distractors["concept"] 
                       + final_distractors["reasoning"] 
                       + final_distractors["visual_interpretation"] 
                       + final_distractors["data_processing"] 
                       + final_distractors["question_bias"])

    user_prompt_fusion = f"""
    Question: {question}
    Correct Answer: {answer}
    All Distractors: {distractors}
    """

    distractors = get_reply(
        client, fusion_generation_system_prompt, user_prompt_fusion, image_base64, Distractors
    )["distractors"]
    
    return distractors


def judge_multichoice_correctness_with_image(
    client, question, choices, answer, image_base64
):
    user_prompt = f"""
    Question: {question}
    Choices: {choices}
    Correct Answer: {answer}
    """
    response = get_reply(
        client,
        evaluator_system_prompt,
        user_prompt,
        image_base64,
        Judgement,
    )
    return response


def improve_multichoice_correctness_with_image(
    client,
    question,
    choices,
    answer,
    issue,
    improvement,
    image_base64,
):
    user_prompt = f"""
    Question: {question}
    Choices: {choices}
    Correct Answer: {answer}
    Identified Issues: {issue}
    Suggested Improvements: {improvement}
    """

    response = get_reply(
        client,
        refiner_system_prompt,
        user_prompt,
        image_base64,
        Question,
    )
    return response


def process_item(client, item, reviewer, refiner):
    
    question = item["question"]
    answer = item["answer"]
    image_base64 = item["image"]


    distactors = convert_to_multi_choice(
        client, question, answer, image_base64, reviewer
    )
    if distractors is None:
        return None
    choices = [item["text"] for item in distactors] + [answer]
    random.shuffle(choices)
    
    if refiner:
        judgement = judge_multichoice_correctness_with_image(
            client, question, choices, answer, image_base64
        )
        if judgement is None:
            return None
        distractors = improve_multichoice_correctness_with_image(
            client,
            question,
            choices,
            answer,
            judgement["reasoning"],
            judgement["improvement"],
            image_base64,
        )
        if distractors is None:
            return None
        choices = distractors["distractors"] + [answer]
        random.shuffle(choices)

    answer_idx = choices.index(answer)
    for idx in range(len(choices)):
        item[chr(65 + idx)] = choices[idx]
    item["answer"] = chr(65 + answer_idx)
    
    # output = f"Question: {question}\n\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n\nAnswer: {'ABCD'[choices.index(answer)]}"
    return item

@click.command()
@click.option("--dataset", type=str, help="Dataset Name", default="PathVQA")
@click.option("--dataset_path", type=str, help="Path to dataset", default=None)
@click.option("--output_path", type=str, help="Path to the output file", default=None)
@click.option("--api_key", type=str, help="OpenAI API key", default=None)
@click.option("--use_reviewer", type=bool, help="Whether to use the reviewer system", default=False)
@click.option("--use_refiner", type=bool, help="Whether to use the refiner system", default=False)

def main(dataset, dataset_path, output_path, api_key, use_reviewer, use_refiner):
    if not dataset_path:
        dataset_path = f"data/{dataset}-500.jsonl"
    data = [json.loads(line) for line in open(dataset_path)]
    # select top 10 data
    data = data[:10]
    if not output_path:
        output_path = f"data/{dataset}-MC.jsonl"
    # filter the data that is already in the output jsonl
    if os.path.exists(output_path):
        output_data = [json.loads(line) for line in open(output_path)]
        output_data_ids = [item["index"] for item in output_data]
        data = [item for item in data if item["index"] not in output_data_ids]
    if not api_key:
        return "Please provide an OpenAI API key"
    client = OpenAI(api_key=api_key)
    # Parallelize using ThreadPoolExecutor
    # with ThreadPoolExecutor(max_workers=128) as executor:
    #     # Submit tasks for parallel execution, associating futures with their indices
    #     futures = [executor.submit(process_item, client, row, use_reviewer, use_refiner) for row in data]
    #     # Create progress bar and track as futures complete
    #     results = []
    #     for future in tqdm(as_completed(futures), total=len(futures)):
    #         results.append(future.result())
    #     # Sort results to maintain the original order
    #     results = [result for result in results if result is not None]
    #     results = sorted(results, key=lambda x: x['index'])
    
    # with open(output_path, "w") as f:
    #     for item in results:
    #         f.write(json.dumps(item) + "\n")
    #         f.flush()
    with ThreadPoolExecutor(max_workers=128) as executor:
        # 打开文件用于边生成边写入
        with open(output_path, "w") as f:
            # 提交任务并使用 tqdm 显示进度
            futures = [executor.submit(process_item, client, row, use_reviewer, use_refiner) for row in data]
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    if future is None:
                        continue
                    result = future.result()
                    if result is not None:
                        # 将结果直接写入文件
                        f.write(json.dumps(result) + "\n")
                        f.flush()  # 确保及时写入文件
                except Exception as e:
                    print(f"Error in processing : {e}")
if __name__ == "__main__":
    main()