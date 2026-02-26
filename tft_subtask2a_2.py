import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import MultiNormalizer, GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import MultiLoss, MAE


def get_text_embeddings(text_list, model_name="bert-base-uncased", batch_size=32):
    """Utility to extract mean-pooled embeddings from a transformer model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i+batch_size]
            encoded = tokenizer(batch_texts, padding=True, truncation=True,
                                max_length=512, return_tensors='pt').to(device)
            outputs = model(**encoded)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)


def main():
    train_csv_path = 'data/train_subtask2a.csv'
    test_csv_path = 'data/test_subtask2.csv'

    df = pd.read_csv(train_csv_path)
    test_users_df = pd.read_csv(test_csv_path)

    test_user_ids = test_users_df['user_id'].unique().tolist()
    df['time_idx'] = df.groupby('user_id').cumcount()

    print("Generating text embeddings...")
    df['text'] = df['text'].fillna("")

    raw_embeddings = get_text_embeddings(df['text'].tolist())

    n_components = 32
    print(f"Reducing embeddings to {n_components} dimensions using PCA...")
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(raw_embeddings)

    emb_cols = [f"text_emb_{i}" for i in range(n_components)]
    emb_df = pd.DataFrame(reduced_embeddings, columns=emb_cols)
    df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)

    train_df = df[~df['user_id'].isin(test_user_ids)].copy()
    inference_df = df[df['user_id'].isin(test_user_ids)].copy()
    train_df['user_id'] = train_df['user_id'].astype(str)
    inference_df['user_id'] = inference_df['user_id'].astype(str)

    max_prediction_length = 1
    max_encoder_length = 20

    training_dataset = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=["valence", "arousal"],
        group_ids=["user_id"],
        min_encoder_length=1,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["valence", "arousal"] + emb_cols,
        target_normalizer=MultiNormalizer([
            GroupNormalizer(groups=["user_id"], transformation=None),
            GroupNormalizer(groups=["user_id"], transformation=None),
        ]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        categorical_encoders={
            "user_id": NaNLabelEncoder(add_nan=True)
        },
    )

    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=64, num_workers=4, persistent_workers=True)

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.03,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=[1, 1],
        loss=MultiLoss([MAE(), MAE()]),
    )

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='auto',
        gradient_clip_val=0.1,
        enable_model_summary=True,
        enable_checkpointing=False,
        logger=False
    )

    print("Training TFT...")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
    )

    inference_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        inference_df,
        predict=True,
        stop_randomization=True
    )

    inference_dataloader = inference_dataset.to_dataloader(
        train=False, batch_size=64, num_workers=4, persistent_workers=True)

    print("Running Inference...")
    prediction_result = tft.predict(
        inference_dataloader, mode="raw", return_x=True)

    raw_predictions = prediction_result[0]
    x = prediction_result[1]

    pred_valence_tensor = raw_predictions.prediction[0]
    pred_arousal_tensor = raw_predictions.prediction[1]

    pred_valence = pred_valence_tensor.squeeze().cpu().numpy()
    pred_arousal = pred_arousal_tensor.squeeze().cpu().numpy()

    prediction_index = inference_dataset.x_to_index(x)
    decoded_user_ids = prediction_index['user_id'].values

    results = pd.DataFrame({
        'user_id': decoded_user_ids,
        'pred_valence': pred_valence,
        'pred_arousal': pred_arousal
    })

    last_values = inference_df.sort_values(
        'time_idx').groupby('user_id').last().reset_index()
    last_values['user_id'] = last_values['user_id'].astype(str)

    final_df = pd.merge(results, last_values[[
                        'user_id', 'valence', 'arousal']], on='user_id', how='left')

    final_df['pred_state_change_valence'] = final_df['pred_valence'] - \
        final_df['valence']
    final_df['pred_state_change_arousal'] = final_df['pred_arousal'] - \
        final_df['arousal']

    output_df = final_df[[
        'user_id', 'pred_state_change_valence', 'pred_state_change_arousal']]

    output_df.to_csv('pred_subtask2a_with_text.csv', index=False)
    print("Done! Predictions saved to pred_subtask2a_with_text.csv")


if __name__ == '__main__':
    main()
