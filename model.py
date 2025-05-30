from typing import Any, List, Dict, Optional
from litellm import acompletion
from pydantic import Field
import weave
import wandb
import time
import asyncio
import os
import PyPDF2

@weave.op
def parse_pdf(pdf_file) -> str:
    """
    Parse a PDF file and return its text content.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


class ChatModel(weave.Model):
    """
    extra ChatModel class to be able to store and version more parameters than just the model name.
    """
    chat_model: str
    cm_temperature: float
    cm_max_new_tokens: int

    def model_post_init(self, __context):
        # either use LiteLLM or local models
        pass

    @weave.op()
    async def predict(self, input: List[Dict[str, str]]) -> dict:
        completion_args = {
            "model": self.chat_model,
            "messages": input,
            "temperature": self.cm_temperature,
            "max_tokens": self.cm_max_new_tokens,
        }
        response = await acompletion(**completion_args)
        # NOTE: make sure that copied values are returned and not references
        return dict(response.choices[0].message)


class AuthoringModel(weave.Model):
    """
    A model that summarizes long text using a chat model and a customizable prompt template.
    """
    chat_model: ChatModel  # Underlying LLM for summarization
    summarization_system_prompt: str  # System-level instructions for summarization
    prompts: List[Dict[str, str]] = Field(default_factory=list)

    def model_post_init(self, __context):
        """
        Initialize the prompt template with system and user-level instructions.
        """
        self.prompts = [
            {"role": "system", "content": self.summarization_system_prompt},
        ]

    @weave.op()
    async def predict(self, input: str) -> dict:
        """
        Summarize a long text using the chat model and the prompt template.
        """

        messages = self.prompts + [{"role": "user", "content": input}]
        completion_args = {
            "model": self.chat_model.chat_model,
            "messages": messages,
            "temperature": self.chat_model.cm_temperature,
            "max_tokens": self.chat_model.cm_max_new_tokens,
        }
        response = await acompletion(**completion_args)

        # NOTE: make sure that copied values are returned and not references
        return dict(response.choices[0].message)["content"]

@weave.op
def predict(
    input: str,
    chat_model: Any,
    system_prompt: str
) -> str:
    authoring_model = AuthoringModel(
        chat_model=chat_model,
        summarization_system_prompt=system_prompt,
                )
    result = asyncio.run(authoring_model.predict(input=input))
    return result

def main(config: dict):
    chat_model = ChatModel(
        name="ChatModel",
        chat_model='gpt-4o-mini',
        cm_max_new_tokens=4096,
        cm_temperature=1,
    )

    authoring_model = AuthoringModel(
        chat_model=chat_model,
        summarization_system_prompt=config["summarization_system_prompt"],
                )

    # You can now directly call authoring_model.predict() with a string:
    user_text = ("The signing of the Declaration of Independence in 1776 marked a radical departure from Britain's colonial rule over the Thirteen Colonies. Drafted mainly by Thomas Jefferson, the document argued that governments exist to protect innate human rights and must govern with the consent of the governed. This text did more than just list grievances against King George III; it placed the fledgling United States on a path toward representative governance rooted in Enlightenment ideals. The effect of the Declaration on world politics was profound. Its bold assertion that a people could cast off a tyrannical ruler influenced countless future movements for self-determination. By openly claiming independence, the founders risked their fortunes and their lives, setting an example that would echo across continents. Over time, the Declaration became a symbol of American identity, a touchstone that citizens would reference and refine as the nation progressed and grappled with its own democratic principles.")

    with wandb.init(project="summarization-app", name="summarization-app-v1") as run:
        result = predict(user_text, chat_model, config["summarization_system_prompt"])
        time.sleep(10)
        run.log({"result": 2})


if __name__ == "__main__":
    weave.init('smle-demo/summarization-app')
    config = {
        'summarization_system_prompt': (
"""You are a clinical documentation specialist tasked with authoring a detailed and structured clinical report. Use clear and precise medical language, following HL7. Ensure the report is comprehensive yet concise, maintaining a professional and factual tone.
Ensure the report adheres to clinical best practices and is structured logically for easy review by healthcare professionals."""        )
    }

    main(config)