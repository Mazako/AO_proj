# Dokumentacja

Szczegółowa dokumentacja projektu jest dostępna pod adresem:
[Projekt na Overleaf](https://www.overleaf.com/project/671cdec1be5005f658edc4f1)

## Konfiguracja GoogleTest

Projekt wykorzystuje GoogleTest do testów jednostkowych. Aby skonfigurować tę bibliotekę, wykonaj poniższe kroki:

1. Sklonuj repozytorium GoogleTest:
   ```bash
   git clone https://github.com/google/googletest.git third_party/googletest


2. Umieść je w katalogu third_party (jako /googletest) w głównym katalogu projektu.

3. Po sklonowaniu, możesz zbudować projekt wraz z testami, korzystając z CMake.
## Uruchamianie testów w różnych IDE

### Uruchamianie testów w CLionie

1. W CLionie przejdź do menu **Run > Edit Configurations...**.
2. Kliknij przycisk **+** i wybierz **Google Test**.
3. W polu **Target** wybierz target testów (np. `miller_rabin_test`).
4. Zapisz konfigurację, nadając jej nazwę, np. `Run All Tests`.
5. Upewnij się, że ta konfiguracja jest aktywna, zanim uruchomisz testy.

### Uruchamianie testów w Visual Studio

1. Upewnij się, że projekt jest otwarty jako **CMake Project**.
2. Visual Studio automatycznie wykryje testy GoogleTest przy odpowiedniej konfiguracji CMake.
3. Testy powinny być dostępne w oknie **Test Explorer**.


### Ogólna konfiguracja dla innych IDE

Jeśli używasz innego IDE, które nie ma wbudowanej obsługi GoogleTest lub CMake, oto kroki, które mogą pomóc w skonfigurowaniu testów:

1. **Dodaj GoogleTest jako podprojekt CMake**: W głównym pliku `CMakeLists.txt` upewnij się, że `add_subdirectory(third_party/googletest)` jest dodane, aby CMake mógł zbudować GoogleTest razem z projektem.

2. **Zdefiniuj target testowy**: W pliku `CMakeLists.txt` projektu dodaj testy jako targety, np.:
   ```cmake
   add_executable(miller_rabin_test tests/miller_rabin_test.cpp)
   target_link_libraries(miller_rabin_test PRIVATE gtest gtest_main)

3. **Zbuduj projekt przy użyciu CMake**: Użyj następujących poleceń w terminalu:
   ```bash
   mkdir -p build
   cd build
   cmake ..
   make

   Po wykonaniu tych kroków, testy powinny być skompilowane jako osobne pliki wykonywalne.

4. **Uruchom testy**: Przejdź do katalogu `build` i uruchom skompilowane pliki wykonywalne dla testów. Przykład:
   ```bash
   ./miller_rabin_test

> ⚠️ **Uwaga**: Jeśli IDE nie wykrywa testów automatycznie, możesz uruchamiać pliki wykonywalne testów bezpośrednio z terminala, jak pokazano powyżej, lub sprawdzić dokumentację IDE, aby zobaczyć, czy obsługuje ręczną integrację z GoogleTest i CMake.
