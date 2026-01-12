import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
df = pd.read_csv('/kaggle/input/traindataset/final_standardized_train_data.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.to(device)
model.eval()
embeddings = []
descriptions = df['Dataset_discription'].tolist()
batch_size = 32

with torch.no_grad():
    for i in tqdm(range(0, len(descriptions), batch_size), desc="Generating Embeddings"):
        batch_descriptions = descriptions[i:i+batch_size]
        inputs = tokenizer(batch_descriptions, return_tensors='pt', padding=True, truncation=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        cls_embedding = last_hidden_states[:, 0, :]
        embeddings.append(cls_embedding.cpu())

embeddings_tensor = torch.cat(embeddings, dim=0)
embeddings_array = embeddings_tensor.numpy()
embedding_df = pd.DataFrame(embeddings_array)
embedding_df.columns = [f'vec{i+1}' for i in range(embedding_df.shape[1])]
df = df.reset_index(drop=True)
final_df = pd.concat([df, embedding_df], axis=1)
final_df = final_df.drop('Dataset_discription', axis=1)

print(final_df.head())

other_columns = [col for col in final_df.columns if 'vec' not in col and col != 'sample_id' and col != 'Unit']
final_df = pd.get_dummies(final_df,columns=['Unit'],drop_first=True)
new_column_order = ['sample_id'] + embedding_df.columns.tolist() + ['Unit_grams'] + ['Unit_ml'] + other_columns

final_df = final_df[new_column_order]
columns_to_convert = ['Unit_grams', 'Unit_ml']
final_df[columns_to_convert] = final_df[columns_to_convert].astype(int)

final_df.to_csv('final_embedded_data.csv', index=False)
