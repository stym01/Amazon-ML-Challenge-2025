# --- SCRIPT 2: HYBRID EXTRACTOR ---
import pandas as pd
import spacy
import re

# --- Load the spaCy model ---
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_lg")
print("Model loaded successfully.")

# --- STEP 1: POPULATE THIS LIST ---
# Use the output from the "Brand Finder" script to add your most common brands here.
# Order matters: place longer names before shorter names (e.g., 'Sour Patch Kids' before 'Sour Patch').
KNOWN_BRANDS = [
    'La Victoria',
    'Bear Creek',
    'Salerno',
    'Sour Patch Kids',
    'Frontier Co-op',
    # --- ADD THE BRANDS YOU FOUND FROM SCRIPT 1 HERE ---
]

def extract_brand_and_category_hybrid(item_name):
    """
    Uses a multi-pass, hybrid approach to find the brand and category.
    """
    brand = None
    category = "Uncategorized"

    if not isinstance(item_name, str):
        return None, category

    # --- Pass 1: Exact Match from KNOWN_BRANDS list (Most Reliable) ---
    for b in KNOWN_BRANDS:
        if re.search(r'\b' + re.escape(b) + r'\b', item_name, re.IGNORECASE):
            brand = b
            break

    doc = nlp(item_name)

    # --- Pass 2: Pattern Matching (If no brand found yet) ---
    # Look for a sequence of capitalized words at the start.
    if not brand:
        match = re.match(r'^([A-Z][a-zA-Z\d\.\-&\'\s]+?)\b', item_name)
        if match:
            # Check if the match is a known entity (ORG, PRODUCT) to increase confidence
            potential_brand = match.group(1).strip()
            # Basic check to avoid overly long matches
            if len(potential_brand.split()) <= 3:
                # Use spaCy to see if this potential brand is recognized as an entity
                brand_doc = nlp(potential_brand)
                if any(ent.label_ in ['ORG', 'PRODUCT'] for ent in brand_doc.ents):
                    brand = potential_brand

    # --- Pass 3: NER as a Fallback (Least Reliable) ---
    if not brand:
        potential_brands = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
        if potential_brands:
            brand = potential_brands[0]

    # --- IMPROVED CATEGORY EXTRACTION using Noun Chunks ---
    title_for_category = item_name
    if brand:
        # Remove the brand to avoid it being the category
        title_for_category = item_name.replace(brand, "").strip()

    cat_doc = nlp(title_for_category)
    # Noun chunks are phrases like "green taco sauce" or "original butter cookies"
    noun_chunks = [chunk.text.lower() for chunk in cat_doc.noun_chunks]

    if noun_chunks:
        # Often the first or second noun chunk is the most descriptive category
        # This logic can be refined, but it's a great starting point
        category = noun_chunks[0].capitalize()
        # Avoid categories that are just numbers or sizes
        if category.replace('.','',1).isdigit():
             if len(noun_chunks) > 1:
                 category = noun_chunks[1].capitalize()
             else:
                 category = "Uncategorized"

    return brand, category

# --- Main Execution Logic ---
print("Loading dataset...")
df = pd.read_csv('first_100_rows.csv')

df['item_name'] = df['catalog_content'].str.extract(r"Item Name: (.*?)\n").squeeze("columns")

print("Extracting Brand and Category with Hybrid model... This will take a while.")
extracted_info = df['item_name'].apply(
    lambda name: pd.Series(extract_brand_and_category_hybrid(name), index=['Brand', 'Category'])
)

print("Assembling final DataFrame...")
final_df = pd.DataFrame({
    'sample_id': df['sample_id'],
    'Brand': extracted_info['Brand'],
    'Category': extracted_info['Category'],
    'Value': df['catalog_content'].str.extract(r"Value: ([\d.]+)").squeeze("columns").astype(float),
    'Unit': df['catalog_content'].str.extract(r"Unit: (.*?)\n").squeeze("columns"),
    'Data_discription': df['catalog_content'],
    'image_link': df['image_link'],
    'price': df['price']
})

final_df['Brand'].fillna('Unknown', inplace=True)

print("\nPreview of HYBRID extraction results:")
print(final_df.head(10))

print("\nSaving results to CSV...")
final_df.to_csv('products_with_hybrid_brands.csv', index=False)
print("\nProcess complete! Check 'products_with_hybrid_brands.csv'.")