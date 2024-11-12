#!/bin/bash

SOURCE="../cmake-build-debug/MillerRabin"
DESTINATION="MillerRabin"

if [ -f "$SOURCE" ]; then
    cp -f "$SOURCE" "$DESTINATION"
    echo "Plik został skopiowany do katalogu 'build'."
else
    echo "Plik $SOURCE nie istnieje."
fi
