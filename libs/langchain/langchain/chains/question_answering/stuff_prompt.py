# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

prompt_template = """You will receive paragraph and additional information. Modify the paragraph to include the additional information.
If no additional information provided return the paragraph with no changes.

{paragraph}

{additional_information}
Helpful Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["paragraph", "additional_information"]
)

system_template = """Use the following additional information to modify the paragraph. If no additional information provided return the paragraph with no changes.
----------------
{additional_information}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{paragraph}"),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)


PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)
