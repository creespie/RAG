from .tokenizer import chunk_splitter, STOPWORDS

def query_search(query, k, retriever, chunks):
    if k <=0:
        raise ValueError(f"In {query} the amount of answers is negative or zero")
    if k > len(chunks):
        k = len(chunks)

    query_words = [word for word in chunk_splitter(query) if word not in STOPWORDS]
    if len(query_words) == 0:
        print (f"No relevant words in {query}")
        return []
    indices, scores = retriever.retrieve([query_words], k=k)
    if scores[0][0] == 0:
        print(f"No matching words in {query}")
        return []
    return [chunks[int(chunk)] for chunk in indices[0]]

if __name__ == "__main__":
    from .indexer import load_index

    chunks, bm = load_index("data/processed")
    print(f"Indice caricato: {len(chunks)} chunk\n")

    cases = [
        ("how to configure lora rank", 5),
        ("what is the", 5),
        ("???", 5),
        ("xyzqwerty", 5),
        ("AsyncLLMEngine", 5),
        ("", 5),
        ("how to configure lora rank", 1000000),
        ("how to configure lora rank", 0),
        ("how to configure lora rank", -3),
    ]

    for query, k in cases:
        print("=" * 60)
        print(f"Domanda: {query!r}   k={k}")
        try:
            results = query_search(query, k, bm, chunks)
        except ValueError as error:
            print(f"  ERRORE GESTITO: {error}")
            continue
        print(f"  {len(results)} risultati")
        for r in results[:5]:
            print(f"  {r.file_path} "
                  f"[{r.first_character_index}:{r.last_character_index}]")
