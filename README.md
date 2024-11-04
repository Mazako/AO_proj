Dokumentacja

https://www.overleaf.com/project/671cdec1be5005f658edc4f1


## Konfiguracja GoogleTest

Projekt wykorzystuje GoogleTest do testów jednostkowych. Aby skonfigurować tę bibliotekę, wykonaj poniższe kroki:

1. Sklonuj repozytorium GoogleTest:
   ```bash
   git clone https://github.com/google/googletest.git third_party/googletest
   
2. Umieść je w katalogu third_party (jako /googletest) w głównym katalogu projektu.

3. Po sklonowaniu, możesz zbudować projekt wraz z testami, korzystając z CMake.