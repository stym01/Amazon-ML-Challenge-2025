import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

# Load your dataset
df = pd.read_csv('final_standardized_train_data.csv')


# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Create a list to store the embeddings
embeddings = []
descriptions = df['Dataset_discription'].tolist()
cnt=0
# Process each description
with torch.no_grad():
    print(cnt)
    cnt+=1
    for description in descriptions:
        inputs = tokenizer(description, return_tensors='pt', padding=True, truncation=True)
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        cls_embedding = last_hidden_states[:, 0, :]
        
        print(cnt)
        cnt+=1
        embeddings.append(cls_embedding)

# Concatenate all embeddings into a single tensor
embeddings_tensor = torch.cat(embeddings, dim=0)

# Convert the tensor to a NumPy array for use in a DataFrame
embeddings_array = embeddings_tensor.numpy()

# Create a DataFrame from the embeddings array
embedding_df = pd.DataFrame(embeddings_array)

# Rename the columns to 'vec1', 'vec2', ...
embedding_df.columns = [f'vec{i+1}' for i in range(embedding_df.shape[1])]

# Reset the index of the original DataFrame to ensure a clean join
df = df.reset_index(drop=True)

# Join the embedding DataFrame with the original DataFrame
final_df = pd.concat([df, embedding_df], axis=1)

# Drop the original 'Dataset_description' column as it's no longer needed
final_df = final_df.drop('Dataset_discription', axis=1)

print(final_df.head())

# --- NEW CODE ADDED HERE ---
# 1. Define the desired column order
other_columns = [col for col in final_df.columns if 'vec' not in col and col != 'sample_id' and col != 'Unit']
final_df = pd.get_dummies(final_df,columns=['Unit'],drop_first=True)
new_column_order = ['sample_id'] + embedding_df.columns.tolist() + ['Unit_grams'] + ['Unit_ml'] + other_columns

# 2. Reorder the DataFrame columns
final_df = final_df[new_column_order]
columns_to_convert = ['Unit_grams', 'Unit_ml']
final_df[columns_to_convert] = final_df[columns_to_convert].astype(int)

# 3. Save the modified DataFrame to a new CSV file
final_df.to_csv('final_embedded_data.csv', index=False)

print("A new CSV file named 'final_rows_with_embeddings.csv' has been created with reordered columns.")