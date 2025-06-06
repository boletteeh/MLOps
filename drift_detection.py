import pandas as pd
import random
import nltk
from nltk.corpus import wordnet, stopwords
import re
import os

# Download NLTK-ressourcer
nltk.download("wordnet")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# ---- Driftfunktioner ---- #

def get_synonym(word):
    """Returner et synonym for et ord, hvis muligt."""
    synonyms = wordnet.synsets(word)
    if synonyms:
        lemmas = synonyms[0].lemmas()
        if len(lemmas) > 1:
            return lemmas[1].name().replace("_", " ")
    return word

def shuffle_words(text):
    """Byt om på to tilfældige naboword."""
    words = text.split()
    if len(words) < 2:
        return text
    idx = random.randint(0, len(words) - 2)
    words[idx], words[idx + 1] = words[idx + 1], words[idx]
    return " ".join(words)

def insert_stopword(text):
    """Tilføj et tilfældigt stopord et sted i teksten."""
    words = text.split()
    pos = random.randint(0, len(words))
    stopword = random.choice(list(stop_words))
    words.insert(pos, stopword)
    return " ".join(words)

def apply_noise(text):
    """Anvend kombineret støj på én tekst."""
    words = re.findall(r'\b\w+\b', text)
    if not words:
        return text
    idx = random.randint(0, len(words) - 1)
    words[idx] = get_synonym(words[idx])
    noisy_text = " ".join(words)
    noisy_text = shuffle_words(noisy_text)
    noisy_text = insert_stopword(noisy_text)
    return noisy_text

# ---- Hovedprogram ---- #

def inject_drift(input_file="test_sent_emo.csv", output_file="test_drifted.csv", text_col="Utterance", drift_frac=0.2):
    if not os.path.exists(input_file):
        print(f"❌ Filen '{input_file}' blev ikke fundet.")
        return

    df = pd.read_csv(input_file)
    if text_col not in df.columns:
        print(f"❌ Kolonnen '{text_col}' findes ikke i filen.")
        return

    noisy_df = df.copy()
    drift_indices = noisy_df.sample(frac=drift_frac, random_state=42).index
    noisy_df.loc[drift_indices, text_col] = noisy_df.loc[drift_indices, text_col].apply(apply_noise)

    noisy_df.to_csv(output_file, index=False)
    print(f"✅ Driftet data gemt som '{output_file}' – {len(drift_indices)} tekster blev modificeret.")


# ---- Kør som script ---- #

if __name__ == "__main__":
    inject_drift()  # Du kan ændre argumenter her

