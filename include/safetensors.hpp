#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <cstdint>
#include <vector>

#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct Tensor
{
    std::string name;
    std::string dtype;

    std::vector<int64_t> shape;

    void* data;
    size_t size;
};

class SafeTensorLoader
{

public:

    ~SafeTensorLoader()
    {
        close();
    }

    bool load(const std::string& path)
    {
        fd = open(
            path.c_str(),
            O_RDONLY
        );

        if (fd < 0)
        {
            std::cerr 
                << "Cannot open safetensors\n";
            return false;
        }

        file_size = lseek(
            fd,
            0,
            SEEK_END
        );

        if (file_size <= 0)
        {
            std::cerr
                << "Invalid file size\n";
            return false;
        }

        // mmap whole file
        mapped = mmap(
            nullptr,
            file_size,
            PROT_READ,
            MAP_PRIVATE,
            fd,
            0
        );

        if (mapped == MAP_FAILED)
        {
            std::cerr
                << "mmap failed\n";

            mapped = nullptr;
            return false;
        }

        // Read header size
        uint64_t header_size;

        memcpy(
            &header_size,
            mapped,
            sizeof(uint64_t)
        );

        // Read JSON header
        char* header_ptr =
            static_cast<char*>(mapped)
            + sizeof(uint64_t);

        std::string header(
            header_ptr,
            header_size
        );
        std::cout << header << std::endl;

        metadata =
            nlohmann::json::parse(header);

        data_offset =
            sizeof(uint64_t)
            + header_size;

        return true;
    }

    void print_tensors()
    {
        for(auto& item : metadata.items())
        {
            std::cout
                << item.key()
                << "\n";


            std::cout
                << "dtype: "
                << item.value()["dtype"]
                << "\n";


            std::cout
                << "shape: "
                << item.value()["shape"]
                << "\n";


            auto start =
                item.value()["data_offsets"][0]
                .get<uint64_t>();


            auto end =
                item.value()["data_offsets"][1]
                .get<uint64_t>();


            std::cout
                << "bytes: "
                << end - start
                << "\n\n";
        }
    }

    // Return pointer directly into mmap
    void* get_tensor(
        const std::string& name
    )
    {
        if (!metadata.contains(name))
        {
            throw std::runtime_error(
                "Tensor not found: " + name
            );
        }
        auto tensor =
            metadata[name];
        uint64_t start =
            tensor["data_offsets"][0]
            .get<uint64_t>();

        return static_cast<char*>(mapped)
            + data_offset
            + start;
    }

    size_t get_tensor_size(
        const std::string& name
    )
    {
        auto tensor =
            metadata[name];
        uint64_t start =
            tensor["data_offsets"][0]
            .get<uint64_t>();
        uint64_t end =
            tensor["data_offsets"][1]
            .get<uint64_t>();


        return end - start;
    }

    std::vector<Tensor> load_all_tensors()
    {
        std::vector<Tensor> tensors;

        for (auto& item : metadata.items())
        {
            if (item.key() == "__metadata__")
            {
                std::cout << item.key();
                continue;
            }
            Tensor tensor;
            tensor.name = item.key();
            tensor.dtype =
                item.value()["dtype"]
                .get<std::string>();
            for (auto& dim : item.value()["shape"])
            {
                tensor.shape.push_back(
                    dim.get<int64_t>()
                );
            }
            uint64_t start =
                item.value()["data_offsets"][0]
                .get<uint64_t>();
            uint64_t end =
                item.value()["data_offsets"][1]
                .get<uint64_t>();
            tensor.size = end - start;
            tensor.data =
                static_cast<char*>(mapped)
                + data_offset
                + start;

            // tensor.data = malloc(tensor.size);
            // memcpy(
            //     tensor.data,
            //     static_cast<char*>(mapped)
            //         + data_offset
            //         + start,
            //     tensor.size
            // );      
                  
            std::cout << "Tensor name: " << tensor.name << std::endl;
            std::cout << "Tensor size: " << tensor.size << std::endl;
            std::cout << "Tensor type: " << tensor.dtype << std::endl;
            tensors.push_back(tensor);
        }
        return tensors;
    }

    nlohmann::json& get_metadata()
    {
        return metadata;
    }

private:
    void close()
    {
        if(mapped)
        {
            munmap(
                mapped,
                file_size
            );

            mapped = nullptr;
        }
        if(fd >= 0)
        {
            ::close(fd);
            fd = -1;
        }
    }
    int fd = -1;
    void* mapped = nullptr;
    size_t file_size = 0;
    uint64_t data_offset = 0;
    nlohmann::json metadata;
};