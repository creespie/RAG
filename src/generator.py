"""Answer generation with a small local language model."""

import re
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from .corpus import read_file
from .models import MinimalSource

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

NO_CONTEXT_ANSWER = ("I could not find any relevant information in the vLLM "
                     "codebase to answer this question.")

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about the vLLM codebase "
    "and its documentation.\n"
    "Rules:\n"
    "- Use ONLY the information in the provided context. Never use your "
    "own knowledge and never guess names, parameters or values.\n"
    "- If the context answers the question, even partially, give that "
    "answer and do not add any disclaimer.\n"
    "- If the context does not contain the answer at all, reply only: "
    "\"I could not find the answer in the provided sources.\"\n"
    "- Answer in at most 3 sentences. Do not copy long code blocks: "
    "quote at most one short command or line of code.\n"
    "- When useful, mention the file the information comes from."
)
def load_model(name: str = DEFAULT_MODEL) -> tuple[Any, Any]:
    """Load the tokenizer and the language model.

    Args:
        name: Hugging Face identifier of the model.

    Returns:
        The tokenizer and the model, ready for generation.
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()
    return tokenizer, model


def source_text(source: MinimalSource, cache: dict[str, str]) -> str | None:
    """Return the text covered by a source, reading its file only once.

    Args:
        source: File path and character range.
        cache: Already read files, by path. Updated in place.

    Returns:
        The text of the range, or None if the file cannot be read.
    """
    if source.file_path not in cache:
        text = read_file(source.file_path)
        if text is None:
            return None
        cache[source.file_path] = text
    return cache[source.file_path][
        source.first_character_index:source.last_character_index]


def build_context(sources: list[MinimalSource], tokenizer: Any,
                  max_tokens: int, cache: dict[str, str]) -> str:
    """Build the context given to the model, best sources first.

    Sources are added in order until the token budget is reached. If even
    the first one does not fit, it is truncated to the budget.

    Args:
        sources: Retrieved sources, best first.
        tokenizer: Tokenizer of the model, used to count tokens.
        max_tokens: Maximum number of context tokens.
        cache: Already read files, by path.

    Returns:
        The context, or an empty string if no source could be read.
    """
    blocks: list[str] = []
    used = 0
    for number, source in enumerate(sources, start=1):
        text = source_text(source, cache)
        if not text or not text.strip():
            continue
        block = f"[Source {number}] {source.file_path}\n{text.strip()}"
        tokens = tokenizer.encode(block, add_special_tokens=False)
        if used + len(tokens) > max_tokens:
            if not blocks:
                blocks.append(tokenizer.decode(tokens[:max_tokens]))
            break
        blocks.append(block)
        used += len(tokens)
    return "\n\n".join(blocks)


def build_messages(question: str, context: str) -> list[dict[str, str]]:
    """Return the chat messages for one question.

    Args:
        question: The user question.
        context: Text of the retrieved sources.

    Returns:
        A system message with the rules and a user message with the
        context followed by the question.
    """
    user = (f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the context above.")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def generate_answer(question: str, sources: list[MinimalSource],
                    tokenizer: Any, model: Any,
                    cache: dict[str, str] | None = None,
                    max_context_tokens: int = 2000,
                    max_new_tokens: int = 256) -> str:
    """Generate an answer grounded in the retrieved sources.

    Args:
        question: The user question.
        sources: Retrieved sources, best first.
        tokenizer: Tokenizer of the model.
        model: The language model.
        cache: Already read files, shared between questions.
        max_context_tokens: Token budget for the context.
        max_new_tokens: Maximum length of the answer, in tokens.

    Returns:
        The answer text.
    """
    if cache is None:
        cache = {}
    context = build_context(sources, tokenizer, max_context_tokens, cache)
    if not context:
        return NO_CONTEXT_ANSWER

    prompt = tokenizer.apply_chat_template(
        build_messages(question, context), tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                            do_sample=True, temperature=0.3, top_p=0.8,
                            top_k=20)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    answer = str(tokenizer.decode(new_tokens, skip_special_tokens=True))
    # Safety net: drop a reasoning block if the model produced one anyway.
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
    return answer.strip() or NO_CONTEXT_ANSWER