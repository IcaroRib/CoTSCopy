for seed in $(seq 0 0); do
  # multivar
  python -u train.py WTH forecast_multivar --alpha 0.0005 --kernels 1 2 4 8 16 32 64 128 --max-train-length 256 --batch-size 16 --archive forecast_csv --repr-dims 320 --max-threads 8 --seed ${seed} --eval
  # univar
  python -u train.py WTH forecast_univar --alpha 0.0005 --kernels 1 2 4 8 16 32 64 128 --max-train-length 256 --batch-size 16 --archive forecast_csv_univar --repr-dims 320 --max-threads 8 --seed ${seed} --eval
done