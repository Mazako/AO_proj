# Instalacja CUDA
Aby zainstalować CUDA, należy najpierw dowiedzieć się, jaką mamy wersję CUDA
wywołując komendę:

```shell
nvidia-smi
```
wersja CUDA powinna być w prawym górnym rogu printa.

Następnie trzeba pobrać z tej strony odpowiednią z wersją instalkę:
[Link](https://developer.nvidia.com/cuda-toolkit-archive)

Wyklikujemy naszą konfigurację, pobieramy instalkę, no i idziemy zgodnie z instalatorem

## Visual Studio
Dobrze mieć zainstalowane na Windzie Visual Studio z setupem do c++. Instaluje się wtedy dużo programów niezbędnych do kompilacji CUDA.
Warto za wczasu dodać zmiennę środowiskową do:
```shell
C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64
```
potrzebuje tego kompilator CUDA.

## Jak sprawdzić czy działa?
Możemy sobie zrobić podstawowy goły program (nazwa pliku: main.cu):

```cpp
#include <stdio.h>

int main() {
    printf("Hello world!");
    return 0;
}
```
Następnie puścić go przez kompilator i odpalić:
```shell
nvcc -o main ./main.cu
./main.exe
```
Jak się kompili i chodzi to super