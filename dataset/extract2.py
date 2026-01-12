import pandas as pd
import re

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

# Load the dataset
# Make sure 'first_100_rows.csv' is in the same directory as the script
df = pd.read_csv('test.csv')

# Apply the function to the 'catalog_content' column
df[['Value', 'Unit', 'Dataset_discription']] = df['catalog_content'].apply(lambda x: pd.Series(extract_attributes(x)))

# Create the final DataFrame with desired columns
df_final = df[['sample_id', 'Dataset_discription', 'Value', 'Unit', 'image_link']]

# Save the transformed data to a new CSV file
df_final.to_csv('final_test_dataset.csv', index=False)

print("Data extraction complete. The new file is saved as 'extracted_data.csv'")
print(df_final.head())hi.puy