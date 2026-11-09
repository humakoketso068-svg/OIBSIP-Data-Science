
import re
from nltk.stem import PorterStemmer

# A standard English stopword list (the same idea as NLTK's list), written out
# here so nobody has to download anything. Apostrophes are removed before the
# check, so "don't" is looked up as "dont".
STOPWORDS = frozenset("""
i me my myself we our ours ourselves you your yours yourself yourselves he him
his himself she her hers herself it its itself they them their theirs themselves
what which who whom this that these those am is are was were be been being have
has had having do does did doing a an the and but if or because as until while of
at by for with about against between into through during before after above below
to from up down in out on off over under again further then once here there when
where why how all any both each few more most other some such no nor not only own
same so than too very s t can will just don should now
im ive id ill youre youve youd youll hes shes thats theyre theyve theyd theyll
weve wed dont doesnt didnt isnt arent wasnt werent hasnt havent hadnt wont wouldnt
shouldnt couldnt cant mustnt neednt aint
""".split())

_stemmer = PorterStemmer()


def tokenize(text):
    """Return a list of (original_word, stem) pairs for the useful words in text."""
    text = str(text).lower()                      # 1. lowercase
    # don't -> dont. The data also holds curly quotes and, because of its old
    # encoding, garbled characters that stand for the same apostrophe: "\x92"
    # and "\x89\u00fb\u00f7" (written in lowercase, because we lowercased above).
    for apostrophe in ("'", "\u2019", "\u2018", "\x91", "\x92", "\x89\u00fb\u00f7"):
        text = text.replace(apostrophe, "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)      # 2. remove punctuation
    pairs = []
    for word in text.split():
        if word in STOPWORDS or len(word) < 2:    # 3. remove stopwords
            continue
        pairs.append((word, _stemmer.stem(word)))  # 4. stem (winning -> win)
    return pairs


def clean_text(text):
    """Return the cleaned message as one string of stems."""
    return " ".join(stem for _, stem in tokenize(text))
