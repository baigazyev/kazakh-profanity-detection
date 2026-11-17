"""Text normalisation shared by training and inference (no learned state)."""
import re
import unicodedata

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\S+@\S+\.\w+")
PHONE_RE = re.compile(r"(?:\+?7|8)[\s\-()]*7\d{2}[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}")
REPEAT_RE = re.compile(r"(.)\1{2,}")                 # 3+ repeats -> 2 ("көттттт" -> "көтт")
NON_TEXT_RE = re.compile(r"[^\w\s\*#@\$]")          # drop punctuation and emoji, KEEP masking symbols * # @ $
SPACE_RE = re.compile(r"\s+")


def mask_pii(text: str) -> str:
    """Replace URLs, e-mails and Kazakh phone numbers with placeholders."""
    text = URL_RE.sub(" <url> ", text)
    text = EMAIL_RE.sub(" <email> ", text)
    return PHONE_RE.sub(" <phone> ", text)


def normalize(text) -> str:
    """lower-case, Unicode NFC, PII placeholders, collapse long letter repeats,
    remove punctuation/emoji but keep the symbols people use to mask swear words."""
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = mask_pii(text)
    text = REPEAT_RE.sub(r"\1\1", text)
    text = NON_TEXT_RE.sub(" ", text)
    text = text.replace("_", " ")
    return SPACE_RE.sub(" ", text).strip()
