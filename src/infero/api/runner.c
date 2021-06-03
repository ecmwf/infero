#include "infero.h"

void run_float() {
    int rank1 = 3;
    int shape1[3] = {2, 2, 3};
    float data1[2 * 2 * 3] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};

    int rank2 = 2;
    int shape2[2] = {2, 3};
    float data2[2 * 3] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    infero_inference_float(data1, rank1, shape1, data2, rank2, shape2);
}

void run_double() {
    int rank1 = 3;
    int shape1[3] = {2, 2, 3};
    double data1[2 * 2 * 3] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};

    int rank2 = 2;
    int shape2[2] = {2, 3};
    double data2[2 * 3] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    infero_inference_double(data1, rank1, shape1, data2, rank2, shape2);
}

int main( ) {
    run_float();
    run_double();
}
