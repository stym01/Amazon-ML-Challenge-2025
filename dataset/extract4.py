import pandas as pd

def extract_attributes(catalog_content):
    """
    Extracts Value, Unit, and Dataset_description from the catalog_content string.
    """
    value = None
    unit = None
    description_parts = []

    if isinstance(catalog_content, str):
        lines = catalog_content.strip().split('\n')
        for line in lines:
            if ':' in line:
                key, val = line.split(':', 1)
                key = key.strip()
                val = val.strip()
                if key.lower() == 'value':
                    value = val
                elif key.lower() == 'unit':
                    unit = val
                else:
                    description_parts.append(val)

    description = ', '.join(description_parts)
    return value, unit, description

def standardize_units(df):
    """
    Standardizes the 'Unit' and 'Value' columns of the DataFrame.
    """
    conversion_map = {
        # Weight units to grams
        'ounce': ('grams', 28.3495), 'oz': ('grams', 28.3495), 'ounces': ('grams', 28.3495),
        'pound': ('grams', 453.592), 'lb': ('grams', 453.592), 'pounds': ('grams', 453.592),
        'gram': ('grams', 1), 'gramm': ('grams', 1), 'grams': ('grams', 1), 'gr': ('grams', 1),
        'kg': ('grams', 1000),

        # Volume units to ml
        'fl oz': ('ml', 29.5735), 'fluid ounce': ('ml', 29.5735), 'fl. oz': ('ml', 29.5735), 
        'fluid ounces': ('ml', 29.5735), 'fluid ounce(s)': ('ml', 29.5735), 'fl ounce': ('ml', 29.5735),
        'millilitre': ('ml', 1), 'milliliter': ('ml', 1), 'ml': ('ml', 1),
        'liters': ('ml', 1000),

        # Count units to 'count'
        'count': ('count', 1), 'ct': ('count', 1), 'each': ('count', 1), 'pack': ('count', 1), 
        'packs': ('count', 1), 'bag': ('count', 1), 'can': ('count', 1), 'jar': ('count', 1), 
        'k-cups': ('count', 1), 'bottle': ('count', 1), 'per carton': ('count', 1), 'piece': ('count', 1), 
        'pouch': ('count', 1), 'per box': ('count', 1), 'per package': ('count', 1), 
        'product_weight': ('count', 1), 'paper cupcake liners': ('count', 1),
        '1': ('count', 1), '8': ('count', 1), 'foot': ('count', 1)
    }

    def convert_row(row):
        unit = row['Unit']
        value = pd.to_numeric(row['Value'], errors='coerce')

        if pd.isna(value) or pd.isna(unit):
            return value, unit

        unit_lower = str(unit).lower()
        
        if unit_lower in conversion_map:
            target_unit, multiplier = conversion_map[unit_lower]
            return value * multiplier, target_unit
        else:
            # If unit is not in map, treat as count
            return value, 'count'

    df[['Value', 'Unit']] = df.apply(convert_row, axis=1, result_type='expand')
    return df

# Step 1: Load and extract attributes from the original CSV
df = pd.read_csv('test.csv')
df[['Value', 'Unit', 'Dataset_discription']] = df['catalog_content'].apply(lambda x: pd.Series(extract_attributes(x)))
df_extracted = df[['sample_id', 'Value', 'Unit', 'Dataset_discription','image_link']]

# Step 2: Standardize the units
df_standardized = standardize_units(df_extracted)

# Save the final DataFrame to a new CSV file
df_standardized.to_csv('final_standardized_test_data.csv', index=False)

print("Unit standardization complete. The new file is saved as 'final_standardized_data.csv'")
print(df_standardized.head())