import os
import typing as t

import openai
from _types import Message, Parameters, Role
from mistralai.client import MistralClient # type: ignore
from mistralai.models.chat_completion import ChatMessage # type: ignore
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


def _chat_openai(
    client: OpenAI,
    messages: t.List[Message],
    parameters: Parameters,
    extra_body: t.Optional[dict] = None,
) -> Message:
    try:
        response = client.chat.completions.create(
            model=parameters.model,
            messages=t.cast(t.List[ChatCompletionMessageParam], messages),
            temperature=parameters.temperature,
            max_tokens=parameters.max_tokens,
            top_p=parameters.top_p,
            extra_body=extra_body,
        )
    except openai.BadRequestError as e:
        # Provider-side content filter (e.g. Azure) — treat as a refusal instead of crashing
        if "content_filter" in f"{e} {e.body}":
            return Message(role=Role.assistant, content="[Blocked by provider content filter]")
        raise

    response_message = response.choices[0].message
    return Message(
        role=Role(response_message.role), content=str(response_message.content)
    )


def chat_openai(messages: t.List[Message], parameters: Parameters) -> Message:
    return _chat_openai(OpenAI(), messages, parameters)


def chat_mistral(
    messages: t.List[Message], parameters: Parameters
) -> Message:
    client = MistralClient()
    messages = [
        ChatMessage(role=message.role, content=message.content) for message in messages
    ]

    response = client.chat(
        model=parameters.model,
        messages=messages,
        temperature=parameters.temperature,
        max_tokens=parameters.max_tokens,
        top_p=parameters.top_p,
    )
    response_message = response.choices[-1].message
    return Message(role=response_message.role, content=response_message.content)

def embed_mistral(contents: t.List[str]) -> t.List[t.List[float]]:
    client = MistralClient()
    response = client.embeddings('mistral-embed', contents)
    return [d.embedding for d in response.data]

def chat_together(messages: t.List[Message], parameters: Parameters) -> Message:
    client = openai.OpenAI(
        api_key=os.environ["TOGETHER_API_KEY"],
        base_url="https://api.together.xyz/v1",
    )

    return _chat_openai(client, messages, parameters)


def _load_dotenv(path: str = os.path.join(os.path.dirname(__file__), ".env")) -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()


def chat_openrouter(messages: t.List[Message], parameters: Parameters) -> Message:
    client = openai.OpenAI(
        api_key=os.environ["OPENAI_COMPATIBLE_API_KEY"],
        base_url=os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "https://openrouter.ai/api/v1"),
    )

    # Skip Azure: it applies its own jailbreak/content filter in front of the model
    return _chat_openai(
        client, messages, parameters, extra_body={"provider": {"ignore": ["Azure"]}}
    )
