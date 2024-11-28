#ifndef FILELOADER_H
#define FILELOADER_H

#include <string>
#include <cstdint>

class FileLoader {
public:
    static std::pair<uint64_t*, int> load_numbers_from_file(const std::string& filename);
};

#endif // FILELOADER_H
