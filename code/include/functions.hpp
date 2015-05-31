#ifndef NEURO_EVOLUTION_FUNCTIONS_HPP
#define NEURO_EVOLUTION_FUNCTIONS_HPP

#include <cmath>
#include <string>

#include <types.hpp>


namespace NeuroEvolution {

/** Sigmoid activation function */
Weight sigmoid(Weight x);

/** Sigmoid activation function derivative */
Weight sigmoid_prime(Weight x);

/** Hyperbolic tangent activation function */
Weight hyperbolic_tangent(Weight x);

/** Hyperbolic tangent activation function derivative */
Weight hyperbolic_tangent_prime(Weight x);


/** Get the string representation of a logistic function */
char const * funcToStr(LogisticFunctionType func);

/** Get the string representation of a logistic function */
char const * funcToStr(LogisticFunction func);

/** Get a logistic function from its string representation */
LogisticFunction strToFunc(std::string const & fname);

}


#endif //NEURO_EVOLUTION_FUNCTIONS_HPP
