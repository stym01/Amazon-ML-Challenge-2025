import pandas as pd
import re

# Load the dataset from the CSV file
df = pd.read_csv('first_100_rows.csv')

def extract_info(content):
    """
    Extracts Brand, Category, Value, Unit, and Data Description
    from the catalog_content column.
    """
    # Initialize dictionary to store extracted data
    data = {
        'Brand': None,
        'Category': None,
        'Value': None,
        'Unit': None,
        'Data_discription': None
    }

    if isinstance(content, str):
        # --- Extract Item Name (used for Brand and Category) ---
        item_name_match = re.search(r"Item Name: (.*?)\n", content)
        item_name = item_name_match.group(1) if item_name_match else ""

        # --- Extract Brand (assuming the first word of the item name) ---
        if item_name:
            data['Brand'] = item_name.split()[0]

            # --- Extract Category (assuming it's part of the item name after the brand) ---
            # This is a simple approach; you might want to refine it based on your specific data
            category_parts = item_name.split(',')[0].split()[1:]
            data['Category'] = ' '.join(category_parts) if category_parts else None


        # --- Extract Value and Unit ---
        value_match = re.search(r"Value: ([\d.]+)", content)
        data['Value'] = float(value_match.group(1)) if value_match else None

        unit_match = re.search(r"Unit: (.*?)\n", content)
        data['Unit'] = unit_match.group(1) if unit_match else None

        # --- Consolidate the description ---
        description_parts = []
        if item_name:
            description_parts.append(f"Item Name: {item_name}")

        bullet_points = re.findall(r"Bullet Point \d+: (.*?)\n", content)
        if bullet_points:
            description_parts.extend(bullet_points)

        product_desc_match = re.search(r"Product Description: (.*?)$", content, re.DOTALL)
        if product_desc_match:
            description_parts.append(product_desc_match.group(1).strip())

        data['Data_discription'] = "\n".join(description_parts)

    return pd.Series(data)

# Apply the function to the 'catalog_content' column
extracted_data = df['catalog_content'].apply(extract_info)

# Create the new DataFrame with the desired columns
new_df = pd.DataFrame({
    'sample_id': df['sample_id'],
    'Brand': extracted_data['Brand'],
    'Category': extracted_data['Category'],
    'Value': extracted_data['Value'],
    'Unit': extracted_data['Unit'],
    'Data_discription': extracted_data['Data_discription'],
    'image_link': df['image_link'],
    'price': df['price']
})

# Display the first few rows of the new DataFrame
print("Preview of the new dataset:")
print(new_df.head())

# Save the new DataFrame to a CSV file
new_df.to_csv('cleaned_products.csv', index=False)

print("\nSuccessfully created 'cleaned_products.csv' with the new format!")