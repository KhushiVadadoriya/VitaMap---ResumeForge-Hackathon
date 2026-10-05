"""
Text preprocessing and cleaning pipeline for VitaMap ResumeForge.
Provides conservative deterministic normalization while strictly preserving technical tokens.
"""

import html
import re
import unicodedata
from typing import Optional, Set

# Protected technical tokens to safeguard during token transformations
PROTECTED_TERMS = {
    "c++": "__TECH_CPP__",
    "c#": "__TECH_CSHARP__",
    ".net": "__TECH_DOTNET__",
    "node.js": "__TECH_NODEJS__",
    "vue.js": "__TECH_VUEJS__",
    "react.js": "__TECH_REACTJS__",
}
INVERSE_PROTECTED_TERMS = {v: k for k, v in PROTECTED_TERMS.items()}

# Standard English stopwords excluding tokens that overlap with technical terms or negation
DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def extract_text_from_html(html_str: Optional[str]) -> str:
    """
    Extract readable plain text from HTML markup using regex and entity unescaping.
    """
    if not html_str or not isinstance(html_str, str):
        return ""
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", html_str)
    # Unescape HTML entities (&nbsp;, &amp;, etc.)
    text = html.unescape(text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract raw text from a PDF file using pdfminer.high_level.
    """
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(pdf_path)
        return text or ""
    except Exception as e:
        return ""


def clean_text_minimal(text: Optional[str]) -> str:
    """
    VARIANT A: Minimal normalization.
    - Handles None/NaN
    - Unescapes HTML entities
    - Removes HTML tags
    - Applies Unicode NFKC normalization (ligatures fi->fi, fullwidth characters)
    - Removes non-printable/control characters (preserving newlines and tabs)
    - Normalizes quotes, dashes, and bullet points
    - Preserves all technical tokens (C++, C#, .NET, SQL, Python, AWS, etc.)
    - Preserves case and section headers
    - Normalizes excessive whitespace and linebreaks
    """
    if text is None or not isinstance(text, str):
        return ""

    # Unescape HTML entities
    text = html.unescape(text)

    # Strip any residual HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Unicode NFKC normalization (resolves ligatures like \ufb01 -> fi, fullwidth hyphens, etc.)
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters except \n and \t
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\xad]", "", text)

    # Normalize bullet points and decorative markers to a clean space
    text = re.sub(r"[\u2022\u25aa\u25c6\u25cf\u25e6\u274f\u2756\u27a2\uf046\uf0a7\uf0b7]", " ", text)

    # Normalize various single quotes, curly apostrophes, and CP1252 artifacts
    text = re.sub(r"[\u2018\u2019\u201a\x92]", "'", text)

    # Normalize double quotes
    text = re.sub(r"[\u201c\u201d\u00ab\u00bb]", '"', text)

    # Normalize dashes and hyphens
    text = re.sub(r"[\u2010\u2013\u2014\u2212\uff0d]", "-", text)

    # Standardize line endings to \n
    text = re.sub(r"\r\n|\r", "\n", text)

    # Collapse excessive linebreaks (> 2 consecutive newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize horizontal whitespace (tabs, multiple spaces, non-breaking spaces)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Strip outer whitespace
    return text.strip()


def clean_text_conservative(text: Optional[str]) -> str:
    """
    VARIANT B: Minimal normalization + conservative token cleanup.
    - Applies clean_text_minimal
    - Protects critical technical tokens (C++, C#, .NET, etc.)
    - Normalizes case to lowercase
    - Normalizes noisy punctuation separators (e.g. repeated dashes, pipes, tildes)
    - Restores protected technical tokens
    """
    text = clean_text_minimal(text)
    if not text:
        return ""

    # Protect technical tokens before lowercasing/punctuation handling
    # Protect C++
    text = re.sub(r"(?:\b|\s)C\+\+(?:\b|\s|[.,;])", lambda m: m.group(0).replace("C++", "TECHCPPTOKEN"), text, flags=re.IGNORECASE)
    # Protect C#
    text = re.sub(r"(?:\b|\s)C\#(?:\b|\s|[.,;])", lambda m: m.group(0).replace("C#", "TECHCSHARPTOKEN"), text, flags=re.IGNORECASE)
    # Protect .NET
    text = re.sub(r"(?:\b|\s)\.NET(?:\b|\s|[.,;])", lambda m: m.group(0).replace(".NET", "TECHDOTNETTOKEN"), text, flags=re.IGNORECASE)
    # Protect Node.js
    text = re.sub(r"(?:\b|\s)Node\.js(?:\b|\s|[.,;])", lambda m: m.group(0).replace("Node.js", "TECHNODEJSTOKEN"), text, flags=re.IGNORECASE)

    # Convert to lowercase
    text = text.lower()

    # Clean excessive noisy punctuation patterns while preserving words and basic sentence punctuation
    # Replace repeated punctuation like ----, ====, ****, ||| with space
    text = re.sub(r"[-=~*_|/\\#]{2,}", " ", text)

    # Normalize whitespace
    text = re.sub(r"[^\S\n]+", " ", text)

    # Restore protected technical tokens to clean lowercase canonical forms
    text = text.replace("techcpptoken", "c++")
    text = text.replace("techcsharptoken", "c#")
    text = text.replace("techdotnettoken", ".net")
    text = text.replace("technodejstoken", "node.js")

    return text.strip()


def clean_text_stopwords(
    text: Optional[str],
    stopwords: Optional[Set[str]] = None
) -> str:
    """
    VARIANT C: Minimal normalization + conservative cleanup + optional stopword removal.
    - Applies clean_text_conservative
    - Removes common English stopwords
    - Strictly preserves technical tokens and domain identifiers
    """
    text = clean_text_conservative(text)
    if not text:
        return ""

    sw = stopwords if stopwords is not None else DEFAULT_STOPWORDS

    tokens = text.split()
    # Filter stopwords, but keep tokens like c++, c#, .net even if short
    filtered = [t for t in tokens if t not in sw or t in {"c++", "c#", ".net"}]
    return " ".join(filtered).strip()


def preprocess_text(text: Optional[str], variant: str = "A") -> str:
    """
    Unified dispatcher for preprocessing variants.

    Parameters:
        text: Raw resume string.
        variant: 'A' (Minimal), 'B' (Conservative token cleanup), 'C' (Stopword removal).
    """
    var = variant.upper().strip()
    if var == "A":
        return clean_text_minimal(text)
    elif var == "B":
        return clean_text_conservative(text)
    elif var == "C":
        return clean_text_stopwords(text)
    else:
        raise ValueError(f"Unknown preprocessing variant: {variant}. Expected 'A', 'B', or 'C'.")


def apply_variant_a(texts):
    """Apply Variant A preprocessing to an iterable of raw texts.

    Deterministic and stateless (no fitting), so it is safe to embed inside
    a pickled sklearn Pipeline. Used by the final inference artifact.
    """
    return [clean_text_minimal(t) for t in texts]
