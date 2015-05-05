#ifndef NEURO_EVOLUTION_FUNCTIONS_HPP
#define NEURO_EVOLUTION_FUNCTIONS_HPP

#include <cmath>

#include <types.hpp>


namespace NeuroEvolution {

/** Sigmoid activation function */
Weight sigmoid(Weight x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

/** Sigmoid activation function derivative */
Weight sigmoid_prime(Weight x)
{
    Weight s = sigmoid(x);
    return s * (1.0 - s);
}

/** Hyperbolic tangent activation function */
Weight hyperbolic_tangent(Weight x)
{
    return std::tanh(x);
}

/** Hyperbolic tangent activation function derivative */
Weight hyperbolic_tangent_prime(Weight x)
{
    return 1.0 - x * x;
}

}

#endif //NEURO_EVOLUTION_FUNCTIONS_HPP
