import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv_path = 'data/train_subtask2b.csv'
    test_csv_path = 'data/test_subtask2.csv'

    train_full = pd.read_csv(train_csv_path)
    test_ids_df = pd.read_csv(test_csv_path)
    test_user_ids = test_ids_df['user_id'].unique().tolist()

    train_df = train_full[~train_full['user_id'].isin(test_user_ids)].copy()
    val_df_raw = train_full[train_full['user_id'].isin(test_user_ids)].copy()

    gold_val = val_df_raw[['user_id', 'disposition_change_valence',
                           'disposition_change_arousal']].drop_duplicates()
    gold_val = gold_val.rename(columns={
        'disposition_change_valence': 'disp_change_valence',
        'disposition_change_arousal': 'disp_change_arousal'
    })
    gold_val.to_csv('gold_val_subtask2b.csv', index=False)

    def process_data(df):
        processed_data = []
        grouped = df.groupby('user_id')

        for user_id, group_df in grouped:
            g1 = group_df[group_df['group'] == 1]
            g2 = group_df[group_df['group'] == 2]

            if g1.empty or g2.empty:
                continue

            g1_text = " [SEP] ".join(g1['text'].astype(str).tolist())

            g1_val_avg = g1['valence'].mean()
            g1_aro_avg = g1['arousal'].mean()

            g2_val_avg = g2['valence'].mean()
            g2_aro_avg = g2['arousal'].mean()

            processed_data.append({
                'user_id': user_id,
                'text': g1_text,
                'input_valence': g1_val_avg,
                'input_arousal': g1_aro_avg,
                'target_valence': g2_val_avg,
                'target_arousal': g2_aro_avg
            })

        return pd.DataFrame(processed_data)

    train_processed = process_data(train_df)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    class TextEmotionDataset(Dataset):
        def __init__(self, data, tokenizer, max_len=512):
            self.data = data
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            row = self.data.iloc[index]
            text = row['text']
            input_scores = torch.tensor(
                [row['input_valence'], row['input_arousal']], dtype=torch.float)
            target_scores = torch.tensor(
                [row['target_valence'], row['target_arousal']], dtype=torch.float)

            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            return {
                'user_id': row['user_id'],
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'input_scores': input_scores,
                'targets': target_scores
            }

    train_dataset = TextEmotionDataset(train_processed, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    class BertBiLSTM(nn.Module):
        def __init__(self):
            super(BertBiLSTM, self).__init__()
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.lstm = nn.LSTM(768, 128, bidirectional=True, batch_first=True)
            self.fc_combine = nn.Linear(256 + 2, 64)
            self.fc_out = nn.Linear(64, 2)
            self.dropout = nn.Dropout(0.1)

        def forward(self, input_ids, attention_mask, input_scores):
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state

            lstm_out, _ = self.lstm(sequence_output)

            lstm_out = lstm_out[:, -1, :]

            combined = torch.cat((lstm_out, input_scores), dim=1)

            x = self.dropout(combined)
            x = torch.relu(self.fc_combine(x))
            x = self.fc_out(x)

            return x

    model = BertBiLSTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.L1Loss()

    model.train()
    for epoch in range(3):
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            input_scores = batch['input_scores'].to(device)
            targets = batch['targets'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, input_scores)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    def process_val_input(df):
        processed_data = []
        grouped = df.groupby('user_id')

        for user_id, group_df in grouped:
            g1 = group_df[group_df['group'] == 1]
            if g1.empty:
                g1 = group_df

            g1_text = " [SEP] ".join(g1['text'].astype(str).tolist())
            g1_val_avg = g1['valence'].mean()
            g1_aro_avg = g1['arousal'].mean()

            processed_data.append({
                'user_id': user_id,
                'text': g1_text,
                'input_valence': g1_val_avg,
                'input_arousal': g1_aro_avg,
                'target_valence': 0.0,
                'target_arousal': 0.0
            })
        return pd.DataFrame(processed_data)

    val_processed = process_val_input(val_df_raw)
    val_dataset = TextEmotionDataset(val_processed, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model.eval()
    pre_val_results = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            input_scores = batch['input_scores'].to(device)
            user_ids = batch['user_id'].tolist()

            preds = model(input_ids, attention_mask, input_scores)
            preds = preds.cpu().numpy()

            input_vals = input_scores.cpu().numpy()

            for i, uid in enumerate(user_ids):
                pred_g2_val = preds[i][0]
                pred_g2_aro = preds[i][1]

                g1_val = input_vals[i][0]
                g1_aro = input_vals[i][1]

                disp_change_val = pred_g2_val - g1_val
                disp_change_aro = pred_g2_aro - g1_aro

                pre_val_results.append({
                    'user_id': uid,
                    'pred_dispo_change_valence': disp_change_val,
                    'pred_dispo_change_arousal': disp_change_aro
                })

    pd.DataFrame(pre_val_results).to_csv('pre_val_subtask2b.csv', index=False)

    def process_test_full_history(df):
        processed_data = []
        grouped = df.groupby('user_id')

        for user_id, group_df in grouped:
            text_concat = " [SEP] ".join(group_df['text'].astype(str).tolist())
            val_avg = group_df['valence'].mean()
            aro_avg = group_df['arousal'].mean()

            processed_data.append({
                'user_id': user_id,
                'text': text_concat,
                'input_valence': val_avg,
                'input_arousal': aro_avg,
                'target_valence': 0.0,
                'target_arousal': 0.0
            })
        return pd.DataFrame(processed_data)

    test_full_processed = process_test_full_history(val_df_raw)
    test_full_dataset = TextEmotionDataset(test_full_processed, tokenizer)
    test_full_loader = DataLoader(
        test_full_dataset, batch_size=8, shuffle=False)

    final_results = []

    with torch.no_grad():
        for batch in test_full_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            input_scores = batch['input_scores'].to(device)
            user_ids = batch['user_id'].tolist()

            preds = model(input_ids, attention_mask, input_scores)
            preds = preds.cpu().numpy()

            input_vals = input_scores.cpu().numpy()

            for i, uid in enumerate(user_ids):
                pred_val = preds[i][0]
                pred_aro = preds[i][1]

                curr_val = input_vals[i][0]
                curr_aro = input_vals[i][1]

                disp_change_val = pred_val - curr_val
                disp_change_aro = pred_aro - curr_aro

                final_results.append({
                    'user_id': uid,
                    'pred_dispo_change_valence': disp_change_val,
                    'pred_dispo_change_arousal': disp_change_aro
                })

    pd.DataFrame(final_results).to_csv('pred_subtask2b.csv', index=False)


if __name__ == '__main__':
    main()
