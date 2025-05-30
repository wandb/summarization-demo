import weave
from weave.scorers import HallucinationFreeScorer, SummarizationScorer
from typing import Any, Optional, List
import json
from openai import OpenAI
import asyncio
from model import AuthoringModel, ChatModel, parse_pdf
from rouge_score import rouge_scorer
import re

# Initialize clients and scorers
llm_client = OpenAI()

hallucination_scorer = HallucinationFreeScorer(
    model_id="openai/gpt-4o",
    column_map={"context": "input", "output": "summary"}
)

summarization_scorer = SummarizationScorer(
    model_id="openai/gpt-4o",
)

@weave.op()
def evaluate_rouge(summary: str, output: dict) -> dict:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(summary.strip().lower(), output.strip().lower())
    rouge_l_fmeasure = scores['rougeL'].fmeasure
    return {'rougeL_fmeasure': rouge_l_fmeasure}


def main(config: dict):

    weave.init('smle-demo/summarization-app')

    evaluation_dataset_v1 = weave.ref('evaluation_dataset:v1').get()
    # data = evaluation_dataset_v1.to_pandas()
    # data = weave.Dataset.from_pandas(df)


    list_of_models = [
        "gpt-4o-mini",
        "gpt-4-turbo",
        'claude-3-5-sonnet-20240620',
        "gpt-3.5-turbo",
        "o1-preview",
        'ft:gpt-4o-2024-08-06:weights-biases::AjD60AbG', # id for finetuned model
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
            dataset=evaluation_dataset_v1,
            scorers=[hallucination_scorer,
                     summarization_scorer,
                     evaluate_rouge],
            name=f"{model}_evaluation",
        )

        asyncio.run(evaluation.evaluate(authoring_model, __weave={"display_name": f"{model}_evaluation"}))

if __name__ == "__main__":

#     config = {
#         'summarization_system_prompt': (
# """You are a clinical documentation specialist tasked with authoring a detailed and structured clinical report. Use clear and precise medical language, following HL7. Ensure the report is comprehensive yet concise, maintaining a professional and factual tone.
# Ensure the report adheres to clinical best practices and is structured logically for easy review by healthcare professionals.
# **Clinical Report**

# **Patient Name:** [Patient's Name]  
# **Age:** [Patient's Age]  
# **Diagnosis:** [Primary Diagnosis]  

# **Summary:**  
# [Brief but structured summary of the patient's condition, treatment, and follow-up plan. Maintain a professional tone and ensure clarity for medical professionals.]
# """        )
#     }

    config = {'summarization_system_prompt': 
              'You are a helpful assistant that summarizes text clearly and concisely. Focus on capturing the key points, main arguments, and essential context. Avoid repetition or unnecessary details. When summarizing technical, academic, or domain-specific content, preserve important terminology and intent. Use a tone appropriate to the subject matter (e.g., professional for business reports, neutral for news, accessible for general content).'}
    main(config)