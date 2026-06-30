from datasets import load_dataset
from collections import defaultdict

ds = load_dataset("Mavkif/Roman-Urdu-Parl-split", split="train")
print(f"Total examples: {len(ds)}")
print(f"Columns: {ds.column_names}")

# ── 1. Search Roman-Urdu column for problematic English loanwords ──────────
# We want to see how the dataset maps these Roman words to Urdu script
roman_keywords = [
    "message", "exercise", "exam", "fail",
    "maafi", "maazi", "dhyan", "badhna",
    "project", "chhod", "delete", "backup"
]

print("\n=== ROMAN-URDU COLUMN SEARCH ===")
for kw in roman_keywords:
    matches = ds.filter(
        lambda x, k=kw: x["Roman-Urdu text"] is not None and
                        k in x["Roman-Urdu text"].lower()
    )
    print(f"\n[{kw}] — {len(matches)} matches")
    for i in range(min(5, len(matches))):
        print(f"  Roman: {matches[i]['Roman-Urdu text']}")
        print(f"  Urdu:  {matches[i]['Urdu text']}")
        print(f"  ---")

# ── 2. Search Urdu column for correct script forms ─────────────────────────
# We want to know if correct transliterations exist in training data at all
urdu_keywords = [
    "میسج",       # message
    "ایکسرسائز",  # exercise
    "ایگزام",     # exam (loanword form)
    "امتحان",     # exam (classical form — checking domain)
    "فیل",        # fail
    "معافی",      # maafi
    "ماضی",       # maazi — checking if it's mislabeled against maafi
    "دھیان",      # dhyan
    "بڑھنا",      # badhna
    "پروجیکٹ",   # project
    "چھوڑ",       # chhod
    "ڈیلیٹ",      # delete
]

print("\n=== URDU COLUMN SEARCH (correct forms) ===")
for kw in urdu_keywords:
    matches = ds.filter(
        lambda x, k=kw: x["Urdu text"] is not None and
                        k in x["Urdu text"]
    )
    print(f"\n[{kw}] — {len(matches)} matches")
    for i in range(min(5, len(matches))):
        print(f"  Roman: {matches[i]['Roman-Urdu text']}")
        print(f"  Urdu:  {matches[i]['Urdu text']}")
        print(f"  ---")

# ── 3. Script consistency check — Arabic ي vs Urdu ی ──────────────────────
arabic_ya = "ي"  # U+064A — Arabic
urdu_ye   = "ی"  # U+06CC — Urdu

arabic_ya_count = ds.filter(
    lambda x: x["Urdu text"] is not None and arabic_ya in x["Urdu text"]
)
print(f"\n=== SCRIPT CONSISTENCY ===")
print(f"Rows with Arabic ي (U+064A) in Urdu column: {len(arabic_ya_count)}")
print(f"Total rows: {len(ds)}")
print(f"Percentage: {100 * len(arabic_ya_count) / len(ds):.2f}%")

# Show a few examples
for i in range(min(5, len(arabic_ya_count))):
    print(f"  Roman: {arabic_ya_count[i]['Roman-Urdu text']}")
    print(f"  Urdu:  {arabic_ya_count[i]['Urdu text']}")
    print(f"  ---")

# ── 4. Tokenizer inspection — what does M2M100 do to loanwords ────────────
from transformers import M2M100Tokenizer

tokenizer = M2M100Tokenizer.from_pretrained("Mavkif/m2m100_rup_tokenizer_both")

words_to_check = ["message", "exercise", "exam", "fail", "maafi", "badhna", "chhod"]
print("\n=== TOKENIZER BREAKDOWN ===")
for w in words_to_check:
    tokens = tokenizer.tokenize(w)
    ids    = tokenizer.encode(w, add_special_tokens=False)
    print(f"  {w:15} → tokens: {tokens}  ids: {ids}")