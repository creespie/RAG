def camel_to_words(st: str) -> list[str]:
    if not st:
        return []
    current = st[0]
    i = 1
    words = []
    while i < len(st):
        if st[i].isupper():
            if i == len(st) - 1:
                if not st[i -1].islower():
                    current += st[i]
                else:
                    words.append(current)
                    current = st[i]
            elif  not st[i + 1].islower() and not st[i -1].islower():
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
    current_words = []
    if current_word == camel_to_words(current_word)[0] and "_" not in current_word:
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
    current_words = []
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
            i+= 1
    if current_word:
        current_words.extend(word_to_sections(current_word))
    return current_words

if __name__ == "__main__":
    print(word_to_sections("daje_Roma_DAJEDAJE"))