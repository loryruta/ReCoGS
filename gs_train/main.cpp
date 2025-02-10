
#include <cstdio>
#include <cstdlib>

int main(int argc, char *argv[]) {
    // Command syntax: <scene> <edit-dataset>
    argc--;
    if (argc != 2) {
        fprintf(stderr, "Invalid syntax: %s <scene> <edit-dataset>\n", argv[0]);
        exit(1);
    }
    argv++;




    return 0;
}
