#include <iostream>
#include "infero_version.h"

int main(int argc, char** argv){

    std::cout << infero_VERSION_MAJOR << "."
              << infero_VERSION_MINOR << "."
              << infero_VERSION_PATCH << std::endl;

    return 0;
}
