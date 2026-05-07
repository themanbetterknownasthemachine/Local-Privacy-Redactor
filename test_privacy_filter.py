from transformers import pipeline

MODEL_PATH = r"C:\models\privacy-filter"

classifier = pipeline(
    "token-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    aggregation_strategy="simple",
)

text = """
Mein Name ist Toni.
Meine Mail ist toni@example.com
Meine Telefonnummer ist +41 79 123 45 67
"""

results = classifier(text)

# Reine Satzzeichen-/Leerzeichen-Treffer entfernen
results = [
    entity
    for entity in results
    if entity["word"].strip() and entity["word"].strip() not in [".", ",", ";", ":"]
]

merged_entities = []

for entity in results:
    start = entity["start"]
    end = entity["end"]
    label = entity["entity_group"]

    if not merged_entities:
        merged_entities.append(entity.copy())
        continue

    last = merged_entities[-1]
    gap = text[last["end"] : start]

    if label == last["entity_group"] and "\n" not in gap and gap.strip() == "":
        last["end"] = max(last["end"], end)
    else:
        merged_entities.append(entity.copy())

redacted_text = text

for entity in sorted(merged_entities, key=lambda x: x["start"], reverse=True):
    start = entity["start"]
    end = entity["end"]
    label = entity["entity_group"].upper()

    redacted_text = redacted_text[:start] + f" [REDACTED_{label}]" + redacted_text[end:]

print("\nORIGINAL:\n")
print(text)

print("\nREDACTED:\n")
print(redacted_text)
