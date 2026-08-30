#include <sys/types.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

int main() {
    pid_t pid = getpid();
    printf("PID: %d\n", (int)pid);

    char stack_buffer[1024 * 1024];
    char *heap_buffer = malloc(1024 * 1024 + 1);

    for (int i = 0; i < 1024 * 1024 + 1; i += 4097){
        heap_buffer[i] = 1;
    }

    printf("stack: %p\n", (void *)stack_buffer);
    printf("heap:  %p\n", (void *)heap_buffer);

    int fd = open("/proc/self/pagemap", O_RDONLY);

    if (fd == -1) {
        perror("open");
        return 1;
    }

    uintptr_t va = (uintptr_t)heap_buffer;
    uintptr_t page_size = sysconf(_SC_PAGESIZE);
    uintptr_t page_va = va & ~(page_size - 1);
    uint64_t vpn = page_va / page_size;
    uint64_t offset = vpn * 8;

    uint64_t entry;

    if (pread(fd, &entry, sizeof(entry), offset) != sizeof(entry)) {
        perror("pread");
        close(fd);
        return 1;
    }

    printf("\nPage information:\n");
    printf("Virtual address : 0x%lx\n", va);
    printf("Page address    : 0x%lx\n", page_va);
    printf("VPN             : 0x%lx\n", vpn);
    printf("Pagemap offset  : 0x%lx\n", offset);
    printf("Pagemap entry   : 0x%016lx\n", entry);

    int present = (entry >> 63) & 1;
    printf("Present         : %d\n", present);

    uint64_t pfn = entry & ((1ULL << 55) - 1);
    printf("PFN             : 0x%lx\n", pfn);

    if (present && pfn != 0)
        printf("Physical address: 0x%lx\n", pfn * page_size);

    close(fd);

    getchar();

    free(heap_buffer);
    return 0;
}