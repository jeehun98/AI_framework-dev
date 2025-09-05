:: 기본 (act/bias 없음, beta=0)
.\bench_regemm 2048 2048 2048 30

:: ReLU + perN bias
.\bench_regemm 2048 2048 2048 30 1.0 0.0 1 1

:: beta=1 (C 읽음) + scalar bias
.\bench_regemm 2048 2048 1024 50 1.0 1.0 0 3
