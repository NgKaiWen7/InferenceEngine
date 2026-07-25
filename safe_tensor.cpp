#include "safetensors.hpp"
#include <stdlib.h>

void print_memory()
{
    FILE *file = fopen("/proc/self/status", "r");

    char line[256];

    while (fgets(line, sizeof(line), file))
    {
        if (strncmp(line, "VmRSS", 5) == 0 ||
            strncmp(line, "VmSize", 6) == 0)
        {
            printf("%s", line);
        }
    }

    fclose(file);
}


int main()
{

    SafeTensorLoader loader1;
    std::vector<Tensor> tensors1;
    loader1.load(
        "models/qwen3-4B/model-00003-of-00003.safetensors"
    );
    tensors1 = loader1.load_all_tensors();


    SafeTensorLoader loader2;
    std::vector<Tensor> tensors2;
    loader2.load(
        "models/qwen3-4B/model-00002-of-00003.safetensors"
    );
    tensors2 = loader2.load_all_tensors();

    SafeTensorLoader loader;
    std::vector<Tensor> tensors;
    loader.load(
        "models/qwen3-4B/model-00001-of-00003.safetensors"
    );
    tensors = loader.load_all_tensors();

    print_memory();
    while(true){};
    return 0;
}