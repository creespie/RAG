"""Turn a text into a list of searchable words."""

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "after",
        "all",
        "also",
        "an",
        "any",
        "are",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "each",
        "explain",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "his",
        "how",
        "i",
        "into",
        "it",
        "its",
        "itself",
        "me",
        "more",
        "most",
        "my",
        "of",
        "on",
        "only",
        "other",
        "our",
        "over",
        "s",
        "she",
        "should",
        "so",
        "some",
        "such",
        "t",
        "tell",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "will",
        "would",
        "you",
        "your",
    }
)


def camel_to_words(st: str) -> list[str]:
    """Split a camelCase or PascalCase word, keeping acronyms together.

    Args:
        st: A word without underscores, e.g. "AsyncLLMEngine".

    Returns:
        Its parts, e.g. ["Async", "LLM", "Engine"].
    """
    if not st:
        return []
    current = st[0]
    i = 1
    words: list[str] = []
    while i < len(st):
        if st[i].isupper():
            if i == len(st) - 1:
                if not st[i - 1].islower():
                    current += st[i]
                else:
                    words.append(current)
                    current = st[i]
            elif not st[i + 1].islower() and not st[i - 1].islower():
                current += st[i]
            else:
                words.append(current)
                current = st[i]
            i += 1
        else:
            current += st[i]
            i += 1
    if current:
        words.append(current)
    return words


def word_to_sections(current_word: str) -> list[str]:
    """Return a word and its snake_case and camelCase parts, lowercased.

    Args:
        current_word: An identifier or a plain word.

    Returns:
        The whole word followed by its parts; a word with nothing to split
        is returned only once.
    """
    current_words: list[str] = []
    single = camel_to_words(current_word)[0] == current_word
    if single and "_" not in current_word:
        current_words.append(current_word)
    elif "_" in current_word:
        current_words.append(current_word)
        for word in current_word.split("_"):
            current_words.extend(camel_to_words(word))
    else:
        current_words.append(current_word)
        current_words.extend(camel_to_words(current_word))
    return [word.lower() for word in current_words]


def chunk_splitter(chunk: str) -> list[str]:
    """Tokenize a text into the words used by BM25.

    Words are runs of letters, digits and underscores; each one is expanded
    with word_to_sections.

    Args:
        chunk: Any text: a chunk of the corpus or a question.

    Returns:
        The words, in order.
    """
    current_words: list[str] = []
    current_word = ""
    i = 0
    while i < len(chunk):
        if chunk[i].isalnum() or chunk[i] == "_":
            current_word += chunk[i]
            i += 1
        elif current_word:
            current_words.extend(word_to_sections(current_word))
            current_word = ""
            i += 1
        else:
            i += 1
    if current_word:
        current_words.extend(word_to_sections(current_word))
    return current_words
