#ifndef FILELOADER_H
#define FILELOADER_H

#include <vector>
#include <string>
#include <cstdint>

class FileLoader {
public:
    static std::vector<uint64_t> load_numbers_from_file(const std::string& filename, int count);
};

#endif // FILELOADER_H
