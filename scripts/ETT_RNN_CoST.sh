alphas=(0.0001 0.00025 0.0005 0.001 0.0025)

for alpha in "${alphas[@]}"; do
  for seed in $(seq 0 0); do
    # multivar
    python -u train.py ETTh1 forecast_multivar --recurrent rnn --alpha "$alpha" --kernels 1 2 4 8 16 32 64 128 --max-train-length 256 --batch-size 32 --archive forecast_csv --repr-dims 320 --max-threads 8 --seed ${seed} --eval
    python -u train.py ETTh2 forecast_multivar --recurrent rnn --alpha "$alpha" --kernels 1 2 4 8 16 32 64 128 --max-train-length 256 --batch-size 32 --archive forecast_csv --repr-dims 320 --max-threads 8 --seed ${seed} --eval
    python -u train.py ETTm1 forecast_multivar --recurrent rnn --alpha "$alpha" --kernels 1 2 4 8 16 32 64 128 --max-train-length 256 --batch-size 32 --archive forecast_csv --repr-dims 320 --max-threads 8 --seed ${seed} --eval
    # univar
    python -u train.py ETTh1 forecast_univar --recurrent rnn --alpha "$alpha" --kernels 1 2 4 8 16 32 64 128 --max-train-length 256 --batch-size 32 --archive forecast_csv_univar --repr-dims 320 --max-threads 8 --seed ${seed} --eval
    python -u train.py ETTh2 forecast_univar --recurrent rnn --alpha "$alpha" --kernels 1 2 4 8 16 32 64 128 --max-train-length 256 --batch-size 32 --archive forecast_csv_univar --repr-dims 320 --max-threads 8 --seed ${seed} --eval
    python -u train.py ETTm1 forecast_univar --recurrent rnn --alpha "$alpha" --kernels 1 2 4 8 16 32 64 128 --max-train-length 256 --batch-size 32 --archive forecast_csv_univar --repr-dims 320 --max-threads 8 --seed ${seed} --eval
  done
done
