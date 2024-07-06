# export CUDA_VISIBLE_DEVICES=5

model_name=TimesWave_boost

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ode/ \
  --data_path ode33.csv \
  --model_id auto312 \
  --model $model_name \
  --data Ode \
  --features M \
  --seq_len 312 \
  --label_len 156 \
  --pred_len 312 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 10 \
  --batch_size 1 \
  --plot_samplerate 10 \
  --draw_samplerate 1 \
  --learning_rate 0.0005

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ode/ \
  --data_path ode33.csv \
  --model_id auto624 \
  --model $model_name \
  --data Ode \
  --features M \
  --seq_len 624 \
  --label_len 312 \
  --pred_len 624 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 10 \
  --batch_size 1 \
  --plot_samplerate 100 \
  --draw_samplerate 1 \
  --learning_rate 0.0005

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ode/ \
  --data_path odevdp.csv \
  --model_id vdp \
  --model $model_name \
  --data Ode \
  --features M \
  --seq_len 1000 \
  --label_len 500 \
  --pred_len 1000 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 2 \
  --dec_in 2 \
  --c_out 2 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 3 \
  --batch_size 1 \
  --plot_samplerate 10 \
  --draw_samplerate 1 \
  --learning_rate 0.0005
