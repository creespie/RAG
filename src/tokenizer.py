# Common English function words: they appear mostly in prose (questions and
# documentation) and say nothing about the topic.
_ENGLISH: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "also", "am", "an",
    "are", "at",
    "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by",
    "can", "could",
    "did", "do", "does", "doing", "down", "during",
    "each",
    "few", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how",
    "i", "into", "it", "its", "itself",
    "just",
    "may", "me", "might", "more", "most", "must", "my", "myself",
    "of", "on", "once", "only", "other", "our", "ours", "ourselves", "over",
    "own",
    "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up",
    "very",
    "was", "we", "were", "what", "when", "where", "which", "who", "whom",
    "why", "will", "would",
    "you", "your", "yours", "yourself", "yourselves",
})

# Pieces of English contractions split by the tokenizer
# ("what's" -> "what", "s"; "don't" -> "don", "t").
_CONTRACTIONS: frozenset[str] = frozenset({
    "s", "t", "d", "ll", "m", "re", "ve", "don", "doesn", "isn", "aren",
    "wasn", "weren", "can", "couldn", "shouldn", "won", "wouldn",
})

# Verbs typical of how questions are phrased, not of what they ask.
_QUESTION_WORDS: frozenset[str] = frozenset({
    "describe", "explain", "tell",
})

# English words that are also Python keywords. Frequent in questions, but
# also in code, where removing them shortens code chunks: test this group
# on its own and drop it if the code recall falls.
_KEYWORD_LIKE: frozenset[str] = frozenset({
    "and", "as", "for", "in", "is", "or", "with",
})

# Words about using things, present in most questions of this corpus.
# Also a separate experiment.
_USAGE: frozenset[str] = frozenset({
    "use", "used", "uses", "using",
})

STOPWORDS: frozenset[str] = (_ENGLISH | _CONTRACTIONS | _QUESTION_WORDS
                             | _KEYWORD_LIKE | _USAGE)


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
    return [word for word in current_words if word not in STOPWORDS]
