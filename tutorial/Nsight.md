# Nsight - zestaw narzędzi do profilowania i debugowania aplikacji GPU

Profilowanie kernelów CUDA za pomocą nsys oraz ncu pomaga zrozumieć, które części kodu wymagają optymalizacji

## Profiler Nsight Systems (nsys)

```
sudo apt install nsight-systems

nsys profile --stats=true ./MillerRabin <filename> <number_count> <M_CPU/S_CPU/GPU/BATCH_GPU> <iterations_per_number>

```

### Przeprowadzona analiza algorytmów (Nsight Systems)

```
sudo nsys profile --stats=true ./MillerRabin ../test_data/liczby.txt GPU 1000

sudo nsys profile --stats=true ./MillerRabin ../test_data/liczby.txt BATCH_GPU 1000
```

### Pliki z wynikami:

1. gpu.qdstrm
2. batch-gpu.qdstrm

## Profiler Nsight Compute (ncu)

```
sudo apt install nsight-compute

ncu ./MillerRabin <filename> <number_count> <M_CPU/S_CPU/GPU/BATCH_GPU> <iterations_per_number>

```

### W oparciu o uzyskane informacje można na przykład:

Zwiększyć lub zmniejszyć threads_per_block,
Dodać więcej __shared__ zmiennych do miller_rabin_kernel,
Eksperymentować z różnymi flagami kompilacji, aby poprawić wydajność, np. -lineinfo dla szczegółowych informacji w Nsight lub --use_fast_math

### Shared Memory:
Można przenieść dane do pamięci współdzielonej w obębie bloku! Jeżeli się używa dużych tablic danych
Pozwala na szybki dostęp do pamięci