import pandas as pd
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import MultiNormalizer, GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import MultiLoss, MAE


def main():
    train_csv_path = 'data/train_subtask2a.csv'
    test_csv_path = 'data/test_subtask2.csv'

    df = pd.read_csv(train_csv_path)
    test_users_df = pd.read_csv(test_csv_path)

    test_user_ids = test_users_df['user_id'].unique().tolist()

    df['time_idx'] = df.groupby('user_id').cumcount()

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
        time_varying_unknown_reals=["valence", "arousal"],
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
        train=True, batch_size=64, num_workers=23, persistent_workers=True)

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=[1, 1],
        loss=MultiLoss([MAE(), MAE()]),
    )

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='auto',
        gradient_clip_val=0.1,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False
    )

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
        train=False, batch_size=64, num_workers=23, persistent_workers=True)

    prediction_result = tft.predict(
        inference_dataloader, mode="raw", return_x=True)

    # The model output (dictionary-like)
    raw_predictions = prediction_result[0]
    x = prediction_result[1]
    # raw_predictions.output["prediction"] is a list: [valence_tensor, arousal_tensor]
    # Each tensor shape: (Batch, Prediction_Length, 1) -> e.g., (64, 1, 1)

    pred_valence_tensor = raw_predictions.prediction[0]
    pred_arousal_tensor = raw_predictions.prediction[1]

    # Squeeze to get shape (Batch,)
    pred_valence = pred_valence_tensor.squeeze().cpu().numpy()
    pred_arousal = pred_arousal_tensor.squeeze().cpu().numpy()

    # Map predictions back to User IDs
    # x_to_index recovers the exact dataframe rows used for these predictions
    prediction_index = inference_dataset.x_to_index(x)
    decoded_user_ids = prediction_index['user_id'].values

    results = pd.DataFrame({
        'user_id': decoded_user_ids,
        'pred_valence': pred_valence,
        'pred_arousal': pred_arousal
    })

    # --- Calculation ---
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

    output_df.to_csv('pred_subtask2a.csv', index=False)


if __name__ == '__main__':
    main()
