#include "yolo.hpp"

int main(int argc, char** argv) {
    Yolo::Utils utils(argc, argv);
    utils.run();

    return 0;
}