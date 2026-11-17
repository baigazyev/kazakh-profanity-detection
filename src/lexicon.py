"""Profanity lexicon used ONLY to audit labels and to find label conflicts.

The classifier never sees this list: it learns from labelled examples.
Patterns are applied to lower-cased text. They are deliberately broad
(spelling variants, transliteration, masking) and every hit was reviewed by hand,
because substrings cause false hits (e.g. "есік" = door, "бесік" = cradle).
"""
import re

# Kazakh obscene / vulgar words and curse formulas (Cyrillic, Latin transliteration, common misspellings)
KAZAKH = [
    r"\b(?:сік|сіг|сик(?!р)|сиг(?!нал)|sik|sig)\w*",      # obscene verb root ("сік-", "сіг-") at word start
    r"\b(?:қотақ|котак|қотак|котақ|qotaq|kotak)\w*",      # obscene noun
    r"\b(?:ам|амы|амың|амыңа|амыңды|амын|ама|am|amy)\b",  # obscene noun (standalone word only)
    r"\b(?:шешең|шешен(?:нің|ди|ді)|шешеңн|sheshe[ńn])\w*",  # "your mother" curse formula
    r"\bәке(?:ң|н)(?:нің|ді)\b",                          # "your father's ..." curse formula
    r"\b(?:жал[аеэ]п|jalap)\w*",                          # vulgar slur (whore)
    r"\b(?:боқ|бок|boq)\b", r"\bбо[қк]қ+\w*", r"\bбоққа\b", r"боққотен",  # vulgar noun (shit), not the verb "боқтау" (= to swear)
    r"\b(?:қотағ|котаг|қотаг|котағ|қотақ)\w*",             # obscene noun, inflected (қотағым ...)
    r"\b(?:көт|кот)(?:ің|іңе|іңді|іне|інен|ен|ке|ті|ік|ски|скиии|тің|ине|ин|іне|н|імізге|теріне|еннен|ен бас)\b",  # vulgar "arse" forms, not "көтер-" (lift)
    r"\bк\.т\w+", r"\bкөттеріне\b", r"\bкөтті\b",
    r"шеше\w*с[іи]г\w*", r"шеше\w*амы\b", r"\bам(?:ын|ынды|ына|ыңа|ыңды|га|ға|н|ң|на)\b",  # "your mother ..." formulas, joined spellings
    r"\b[шщ]{2,}[сc]\b",                                  # abbreviation of the "your mother" obscene formula
    r"\bдал[бп][аоы][йи]?[еёоя]+[бп]\w*",                  # Kazakh spellings of Russian "долбоёб" (not "далбаса" = fool)
    r"\b[кқ]аншы[кқ]\w*",                                 # vulgar slur (bitch)
    r"\bта[шс]а[кқғг]\w*",                                # vulgar anatomical
]

# Russian obscene / vulgar words (often inside Kazakh comments), incl. spelling distortions
RUSSIAN = [
    r"(?:\b|на|по|а|о|ни|не)ху[йеёия]\w*", r"пизд\w*", r"\bбля\w*", r"\bбл[еэә]+[тд]\w*",
    r"\b(?:за|у|вы|от|раз)?е[бп]а[лнт]\w*", r"\b(?:за|до)?[её]б\w*", r"заеб\w*", r"зайбал\w*", r"\bеба\b",
    r"\bсу(?:ка|ки|кa|чк|чар)\w*", r"мудак\w*", r"пид[оа]?р\w*", r"залуп\w*", r"шлюх\w*",
    r"гандон\w*", r"долбо\w*", r"говн\w*", r"дерьм\w*", r"\bжоп\w*",
    r"\b(?:huy|hui|pizd|blya|blyat|ebat|suka)\w*",
    r"гавн\w*", r"\bпздц\w*", r"зайеб\w*", r"\bзаиб\w*", r"\bсукі\w*", r"\bбл[әэа]{2,}\w*", r"\bбл[әэ]\b",
    r"\bзб\b", r"нах+у\w*", r"\b(?:фак|fuck)\w*",
]

# Masked forms such as "б**", "х*й", "п***ц"
MASKED = [r"\b[бхпсеё][\*\#@]{1,}[а-яәіңғүұқөһ]*", r"\w*\*{2,}\w*"]

PROFANE_RE = re.compile("|".join(KAZAKH + RUSSIAN + MASKED))

# Insults WITHOUT profanity: label 0 under our rules (see README)
INSULT_RE = re.compile(
    r"\b(?:мал|малдар|малсың|малдарың|есек|ешак|ешек|тупой|тупые|ақымақ|акымак|дурак|идиот|"
    r"далбасагер\w*|қатын\w*|катын\w*|ит|иттер)\b"
)


def profane_hits(text: str):
    return [m.group(0) for m in PROFANE_RE.finditer(str(text).lower())]


def insult_hits(text: str):
    return [m.group(0) for m in INSULT_RE.finditer(str(text).lower())]
