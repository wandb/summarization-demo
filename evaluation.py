import weave
from weave.scorers import HallucinationFreeScorer, SummarizationScorer
from typing import Any, Optional, List
import json
from openai import OpenAI
import asyncio
from model import AuthoringModel, ChatModel, parse_pdf
from rouge_score import rouge_scorer
import re
import pandas as pd
from dotenv import load_dotenv
import os

@weave.op()
def evaluate_rouge(summary: str, output: dict) -> dict:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(summary.strip().lower(), output.strip().lower())
    rouge_l_fmeasure = scores['rougeL'].fmeasure
    return {'rougeL_fmeasure': rouge_l_fmeasure}


def main(config: dict):

    hallucination_scorer = HallucinationFreeScorer(
        model_id="openai/gpt-4o",
        column_map={"context": "input", "output": "summary"}
    )

    summarization_scorer = SummarizationScorer(
        model_id="openai/gpt-4o",
    )

    weave.init(f'{os.getenv("WANDB_ENTITY")}/summarization-app')

    df = pd.read_json('dataset.json')
    dataset = weave.Dataset.from_pandas(df)

    list_of_models = [
        "gpt-4o-mini",
        "gpt-4-turbo",
        'claude-3-5-sonnet-20240620',
        "gpt-3.5-turbo",
        "o1-preview",
        # 'ft:gpt-4o-2024-08-06:weights-biases::AjD60AbG', # id for a finetuned model in openai
    ]

    for model in list_of_models:
        chat_model = ChatModel(
            name=f'{model}',
            chat_model=model,
            cm_max_new_tokens=4096,
        cm_temperature=1,
        )

        authoring_model = AuthoringModel(
        chat_model=chat_model,
            summarization_system_prompt=config["summarization_system_prompt"],
        )
        
        evaluation = weave.Evaluation(
            dataset=dataset,
            scorers=[hallucination_scorer,
                     summarization_scorer,
                     evaluate_rouge],
            name=f"{model}_evaluation",
        )

        asyncio.run(evaluation.evaluate(authoring_model, __weave={"display_name": f"{model}_evaluation"}))

if __name__ == "__main__":
    load_dotenv(".env")

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
    os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")

    config = {
        'summarization_system_prompt': (
"""You are a clinical documentation specialist tasked with authoring a detailed and structured clinical report. Use clear and precise medical language, following HL7. Ensure the report is comprehensive yet concise, maintaining a professional and factual tone.
Ensure the report adheres to clinical best practices and is structured logically for easy review by healthcare professionals.
**Clinical Report**

**Patient Name:** [Patient's Name]  
**Age:** [Patient's Age]  
**Diagnosis:** [Primary Diagnosis]  

**Summary:**  
[Brief but structured summary of the patient's condition, treatment, and follow-up plan. Maintain a professional tone and ensure clarity for medical professionals.]
"""        )
    }

    main(config)