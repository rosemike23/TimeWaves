# export CUDA_VISIBLE_DEVICES=5

model_name=TimesWave_boost

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
  --d_model 32 \
  --d_ff 32 \
  --top_k 1 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 5 \
  --batch_size 32 \
  --plot_samplerate 20 \
  --draw_samplerate 1 \
  --learning_rate 0.0008


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
  --d_model 32 \
  --d_ff 32 \
  --top_k 2 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 5 \
  --batch_size 32 \
  --plot_samplerate 20 \
  --draw_samplerate 1 \
  --learning_rate 0.0008

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
  --d_model 32 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 5 \
  --batch_size 32 \
  --plot_samplerate 20 \
  --draw_samplerate 1 \
  --learning_rate 0.0008

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
  --d_model 32 \
  --d_ff 32 \
  --top_k 4 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 5 \
  --batch_size 32 \
  --plot_samplerate 20 \
  --draw_samplerate 1 \
  --learning_rate 0.0008


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
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 5 \
  --batch_size 32 \
  --plot_samplerate 20 \
  --draw_samplerate 1 \
  --learning_rate 0.0008