/**
 * 1. initialize - random
 * 2. feedforward matops 
 * 3. optimized sigmoid
 * 4. cost function 
 * 5. you can see how SIMD can be used for a lot of this so put it everywhere
 */

#include <stdio.h>
#include <iostream>
#include <immintrin.h>
#include <chrono>
#include <math.h>
#include "..\voltage\voltage.hpp"
using namespace std;
using namespace std::chrono;
const float E = 2.71828182845904523536;

float sigmoid(const float& fl){
       //naive - vectorize this
       return (1/(1+pow(E, -1*fl)));
};

void initialize_weights(){
    //what data type?
};
void feedforward();

void backprop();