# Co to jest CUDA?
CUDA (Compute Unified Device Architecture) to platforma obliczeniowa oraz model programowania stworzony przez firmę NVIDIA, który umożliwia wykonywanie obliczeń równoległych na procesorach graficznych (GPU). 

## Czym się różni od CPU?
Ma o wiele więcej wątków niż procesor. Pozwala to na wykonywanie nawet tysięcy operacji jednocześnie.

# Architektura cuda
Są dwa rodzaje przestrzeni w CUDA:
- Host - jest to inaczej CPU. Używa pamięci RAM
- Device - jest to GPU. Używa pamięci VRAM

## Jak zorganizowane sa rdzenie CUDA?
1. Thread - pojedyńczy wątek, na którym wykonywane są obliecznia.
    - Każdy wątek ma swoją lokalną pamięć, która nie jest współdzielona
2. Wątki są połączone w Bloki (Blocks)
    - jednostką pomocniczą w blokach są warpy
      - Składają sie na 32 wątki w ramach bloku, które wykonują tę samą instrukcje na GPU w jednym cyklu zegara
      - blok może składać się z wielu warpów
    - każdy blok ma pamięć współdzieloną, którą widzą wszystkie składowe wątki 
3. Bloki są połączone w siatki (Grid)
    - Siatka ma dostęp do całej pamięci VRAM
5. Cały program CUDA (kernel) jest odpalany na gridzie.

Tak wiec jeżeli mamy jakiś problem do zrównoleglenia, możemy zbudować sobie całą siatkę, podzielić je na podproblemy w gridzie, i dochodzić do rozwiązań na pojedyńczych wątkach.

Pomysł bo mi ucieknie:
- Siatka: różne liczby [N] dla testu millera rabina
- Grid: pojedyńcza liczba N dla wielu świadków
- Wątek: test dla jednego N i jednego świadka
  
### Jak odnosić się do grida, siatki i wątku w kodzie?
- `gridDim` - zmienna wskazująca liczbę (wymiar) bloków w gridzie
- `blockIdx` - indeks bloku w gridzie
- `blockDim` - liczba (wymiar) wątków w bloku
- `threadIdx` - indeks wątku w bloku

możemy używać *.x, y, z do określenia wymiaru

Jeżeli mamy jakiś problem np z tablicą jednowymiarową, i podzielimy go sobie na 
podproblemy w gridach i wątkach, to żeby znaleźć indeks tablicy na jakiej wątek ma zrobić operacje możemy użyć wzoru:
```
i = blockIdx.x * blockDim.x + threadId.x
```

Bierzemy identyfikator bloku, mnożymy go razy ilość wątków w jednym gridzie i dodajemy identyfikator wątku

wtedy mamy pewność, że jeżeli zaalokowalismy odpowiednią ilość wątków i gridów, to każdy wątek wykona operacje na jednym indeksie tablicy


## Rodzaje funkcji
- `__global__` - I CPU i GPU mogą taką funkcję wywołać
- `__device__` - Tylko GPU może wywołać
- `__host__` - Tylko CPU (można po prostu tego nie pisać i będzie się tak zachowywać)

## Zarządzanie pamięcią
- `CudaMalloc` - alokowanie pamięci na VRAM
- `CudaMemcpyHostToDevice` - kopiowanie pamięci z CPU na GPu
- `CudaMemcpyDeviceToHost` - kopiowanie pamięci z GPU na CPU
- `CudaMemcpyDeviceToDevice` - kopiowanie pamięci z GPU na GPU (z różnych miejsc)
- `CudaFree` - czyszczenie pamięci na GPU

## Jak odaplić kernel?
Po prostu w fukcji hosta wywołujemy ją z parametrami:
```
funkcja<<<LICZBA_BLOKOW, LICZBA_WATKOW_W_BLOKU, NS, STREAM>>>(parametry)
```

Podane liczby mogą być zwykłego typu `int` - wtedy alokujemy tylko jeden wymiar `x`, albo specjalnego dla CUDA typu `dim3` - który jest wymiarem trójwymiarowym. Wtedy mamy `x`, `y`, `z`.

NS to rozmiar współdzielonej pamięci w bajtach, która powinna być dynamicznie zaalokowana na każdy blok (zazwyczaj pomijane, 0 lub nic)

do STREAM wrócimy później

## Bezpieczeństwo wątków
Główną funkcją do bezpieczeństwa wątków jest `cudaDeviceSynchronize()`. Zapewnia ona, że wszystkie wątki funkcji na GPU skończą działanie, zanim znowu wrócimy do CPU. Bardzo ważne żeby używać tej funkcji, zanim pobierzemy jakiekolwiek wyniki z GPU.

Dodatkową funkcją którą warto zapamiętać (już wywoływaną na GPU), jest `__syncthreads`. Zapewnia ona, że do tego momentu wszystkie wątki w bloku wykonają do tego momentu instrukcje. Przydatne jeżeli używamy pamięci współdzielonej, albo program działa tak z tablicą, że potrzebuje policzonych elementów innych indeksów niż tylko swojego.

Przykład? Równoległy algorytm przesunięcia elementów tablicy o 1 w prawo. Wątek musi pobrać wartość swojego lewego sąsiada, a następnie wrzucić tę wartość na miejsce swojego indeksu. Bez `__syncthreads` po instrukcji pobierania sąsiada, i po instrukcji przypisania go program może dawać nieoczekiwane wyniki.

`__syncwarps` działa podobnie na poziomie warpów

## Operacje atomowe
Może się zdarzyć, że wiele wątków będzie jednocześnie wykonywać operacje na jednej zmiennej. Potrzebujemy wtedy pewnej synchronizacji, żeby wartości były zgodne z oczekiwaniami

Na przykład mając program, który w wielu wątkach inkrementuje jedną zmienną, bez synchronizacji wyniki będą złe.

w funkcjach CUDA możemy wywoływać w takich przypadkach funkcje, które zapewnią że dana zmienna w pamięci VRAM jest "zalockowana" i nie może być używana przez inne zmienne.

przykłady funkcji:

- `atomicAdd(T* addr, T v)` - dodaje bezpiecznie wartość `v` do zmiennej addr
- `atomicSub(T* addr, T v)` - podobnie, ale z dodawaniem
- `atomicExch(T* addr, T v)` - podmienia wartość na inną
- [I wiele więcej...](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions)


## Streamy
Streamy są używane do wykonywania asynchronicznych operacji. Definiujemy je typem `cudaStream_t`, i tworzymy instrukcją `cudaStreamCreate(&stream)`. 

Możemy wtedy wykonywać pewnie operacje na strumieniach asynchronicznie. Przykładowo ładować do pamięci GPU instrukcją `cudaMemcpyAsync`. Mamy wtedy uzysk czasowy.

Przykład:
- Ładujemy 1 tablice do strumienia 1
- Ładujemy 2 tablice do strumienia 2
- Puszczamy kernel na strumieniu 1
- Pobieramy wynik w strumienu 1

Wtedy operacje ładowania wykonują się jednocześnie. Kernel może się wykonywać jednocześnie z ładowaniem do pamięci 2 tablicy. Operacje mogą na siebie nachodzić, zamiast w tradycyjnym podejściu blokować się i czekać na zakończenie poprzedniej.

Musimy pamiętać żeby streamy synchronizować instrukcją `cudaStreamSynchronize(stream)`

### Zaawansowane streamy

Dla lepszej kontroli nad pamięcią, możemy zamiast `malloc` używać funkcji `cudaMallocHost`. Zapewnia ona, że zaalokowana pamięć w RAMie nie zmieni swojego adresu przez różne mechanizmy systemu operacyjnego. Aby taką pamięć zwolnić używamy funkcji `CudaFreeHost`

Strumienie można również priorytetyzować. Stream z większym priorytetem wykona się szybciej, jeżeli GPU będzie miało jakieś wolne zasoby:
```cpp
cudaStream_t stream1, stream2;
int leastPriority, greatestPriority;
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, leastPriority);
cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, greatestPriority);
```
Na tym przykładzie tworzymy dwa streamy, przy czym stream2 będzie miał większy priorytet niż stream1


Strumienie można również synchronizować eventami. Możemy stworzyć jakiś event, zapisać go po jakiś operacjach, a następnie pozwolić innemu streamowi na działanie dopiero, jak event w jednym streamie się pojawi.


```cpp
cudaEvent_t event;
cudaEventCreate(&event);

cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream1);
kernel1<<<(..., stream1>>>(d_data, N);

cudaEventRecord(event, stream1);

cudaStreamWaitEvent(stream2, event, 0);

kernel2<<<..., stream2>>>(d_data, N);
```

W tym przykładzie stworzony event, zostanie wyemitowany w momencie, jak pamięć zostanie wczytana, oraz wykona sie kernel1. instrukcją `cudaStreamWaitEvent` każemy streamowi 2 czekać, dopóki event nie zostanie wyemitowany. Dopiero wtedy możemy uruchomić kernel2


Możemy też dawać callbacki do streamów, czyli obsługę tego co się stanie po wykonaniu kernela wdanym streamie


```cpp
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    printf("Skończyłem\n");
}

kernel<<<grid, block, 0, stream>>>(args);
cudaStreamAddCallback(stream, MyCallback, nullptr, 0);
```