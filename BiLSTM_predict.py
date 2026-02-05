import argparse
from pathlib import Path

import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertModel, DistilBertTokenizer

MODEL_NAME = "distilbert-base-uncased"
MAX_SEQUENCE_LENGTH = 128
MAX_TOKEN_LENGTH = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DISTILBERT_HIDDEN_DIM = 768
BATCH_SIZE = 8


class DatasetSubtask1(torch.utils.data.Dataset):
    def __init__(self, df: pl.DataFrame):
        # Sort by user and time
        df = df.sort(["user_id", "timestamp"])
        
        self.processed_df = df.group_by('user_id').agg([
            pl.col('text_id').implode().alias('text_ids'),
            pl.col('text').implode().alias('texts')
        ])
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        self.embedder = DistilBertModel.from_pretrained(MODEL_NAME, device_map=DEVICE)
        self.embedder.eval()

    def __len__(self):
        return self.processed_df.height

    def __getitem__(self, index):
        row = self.processed_df.row(index, named=True)
        
        text_embeddings = self.encode_texts(row['texts'])
        current_seq_len = text_embeddings.size(0)

        pad_len = MAX_SEQUENCE_LENGTH - current_seq_len
        
        padding_emb = torch.zeros(pad_len, text_embeddings.size(1), dtype=torch.float32)
        padded_embeddings = torch.cat([text_embeddings, padding_emb], dim=0)
        
        mask = torch.ones(current_seq_len, dtype=torch.bool)
        mask_padding = torch.zeros(pad_len, dtype=torch.bool)
        full_mask = torch.cat([mask, mask_padding])

        return {
            "embeddings": padded_embeddings,
            "mask": full_mask,
            "user_id": row['user_id'],
            "text_ids": row['text_ids'][:current_seq_len]
        }

    @torch.no_grad()
    def encode_texts(self, texts):
        texts = texts[:MAX_SEQUENCE_LENGTH]
        embeddings_list = []
        for i in range(0, len(texts), 32):
            batch_texts = texts[i:i+32]
            encoded = self.tokenizer(batch_texts, padding='max_length', truncation=True, 
                                     max_length=MAX_TOKEN_LENGTH, return_tensors='pt')
            input_ids = encoded['input_ids'].to(DEVICE)
            mask = encoded['attention_mask'].to(DEVICE)
            output = self.embedder(input_ids=input_ids, attention_mask=mask)
            embeddings_list.append(output.last_hidden_state[:, 0, :].cpu())
        
        if len(embeddings_list) > 0:
            return torch.cat(embeddings_list, dim=0)
        else:
            return torch.empty(0, 768)


class ModelSubtask1(nn.Module):
    def __init__(self, lstm_hidden_dim=256, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=DISTILBERT_HIDDEN_DIM,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            bidirectional=True, 
            batch_first=True
        )
        self.dropout = nn.Dropout(0.3)
        self.regressor = nn.Linear(lstm_hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        return self.regressor(out).squeeze(-1)


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length text_ids
    """
    embeddings = torch.stack([item['embeddings'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    user_ids = [item['user_id'] for item in batch]
    text_ids = [item['text_ids'] for item in batch]
    
    return {
        'embeddings': embeddings,
        'mask': masks,
        'user_id': user_ids,
        'text_ids': text_ids
    }


def predict(model, loader):
    """
    Generate predictions
    """
    model.eval()
    
    all_predictions = []
    all_text_ids = []
    all_user_ids = []
    
    with torch.no_grad():
        for batch in loader:
            emb = batch['embeddings'].to(DEVICE)
            preds = model(emb)
            masks = batch['mask']
            
            # Process each sample in the batch
            for i in range(preds.size(0)):
                user_preds = preds[i][masks[i]].cpu().numpy()
                text_ids = batch['text_ids'][i]
                user_id = batch['user_id'][i]
                
                for pred, text_id in zip(user_preds, text_ids):
                    all_predictions.append(float(pred))
                    all_text_ids.append(int(text_id))
                    all_user_ids.append(int(user_id))
    
    return all_user_ids, all_text_ids, all_predictions


def main():
    parser = argparse.ArgumentParser(description="Predict using trained LSTM models")
    parser.add_argument('--valence_model', type=str, default='models/model_task1_valence.pt',
                       help='Path to trained valence model')
    parser.add_argument('--arousal_model', type=str, default='models/model_task1_arousal.pt',
                       help='Path to trained arousal model')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input data (CSV)')
    parser.add_argument('--output', type=str, default='output/predictions.csv',
                       help='Path to save predictions')
    args = parser.parse_args()

    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Load data
    df = pl.read_csv(args.input)
    if "timestamp" in df.columns:
        df = df.with_columns(pl.col("timestamp").str.to_datetime())
    
    print(f"Loaded {df.height} samples from {args.input}")
    print(f"Number of users: {df['user_id'].n_unique()}")
    
    # Create dataset
    dataset = DatasetSubtask1(df)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    
    # Process valence
    print("\n" + "="*60)
    print("PROCESSING VALENCE")
    print("="*60)
    
    valence_model = ModelSubtask1().to(DEVICE)
    valence_model.load_state_dict(torch.load(args.valence_model, map_location=DEVICE))
    print(f"Loaded valence model from {args.valence_model}")
    
    user_ids, text_ids, valence_preds = predict(valence_model, loader)
    print(f"Generated {len(valence_preds)} valence predictions")
    
    # Process arousal
    print("\n" + "="*60)
    print("PROCESSING AROUSAL")
    print("="*60)
    
    arousal_model = ModelSubtask1().to(DEVICE)
    arousal_model.load_state_dict(torch.load(args.arousal_model, map_location=DEVICE))
    print(f"Loaded arousal model from {args.arousal_model}")
    
    _, _, arousal_preds = predict(arousal_model, loader)
    print(f"Generated {len(arousal_preds)} arousal predictions")
    
    # Create output dataframe
    print("\n" + "="*60)
    print("CREATING OUTPUT")
    print("="*60)
    
    output_df = pl.DataFrame({
        'user_id': user_ids,
        'text_id': text_ids,
        'pred_valence': valence_preds,
        'pred_arousal': arousal_preds
    })
    
    # Save output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output_df.write_csv(args.output)
    
    print(f"\nPredictions saved to {args.output}")
    print(f"Output contains {output_df.height} rows")
    print(f"First few rows:")
    print(output_df.head())
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == '__main__':
    main()