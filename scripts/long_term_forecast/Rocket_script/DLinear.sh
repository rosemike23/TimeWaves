# export CUDA_VISIBLE_DEVICES=2

model_name=DLinear

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/rocket/ \
  --data_path col1.csv \
  --model_id rocket_pred \
  --model $model_name \
  --data Rocket \
  --features M \
  --seq_len 300 \
  --label_len 150 \
  --pred_len 75 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 1 \
  --batch_size 4 \
  --plot_samplerate 5 \
  --draw_samplerate 1 \
  --learning_rate 0.0008