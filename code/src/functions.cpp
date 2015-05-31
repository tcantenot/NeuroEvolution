#include <functions.hpp>


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


/** Get the string representation of a logistic function */
char const * funcToStr(LogisticFunctionType func)
{
    if(func == sigmoid)
    {
        return "SIGMOID";
    }
    else if(func == sigmoid_prime)
    {
        return "SIGMOID_PRIME";
    }
    else if(func == hyperbolic_tangent)
    {
        return "HYPERBOLIC_TANGENT";
    }
    else if(func == hyperbolic_tangent)
    {
        return "HYPERBOLIC_TANGENT_PRIME";
    }
    else
    {
        return "UNKNOWN_FUNCTION";
    }
}

char const * funcToStr(LogisticFunction func)
{
    LogisticFunctionType const * fp = func.target<LogisticFunctionType>();
    if(fp)
    {
        return funcToStr(*fp);
    }
    else
    {
        return "UNKNOWN_FUNCTION";
    }
}

LogisticFunction strToFunc(std::string const & fname)
{
    LogisticFunction func;

    if(fname == "SIGMOID")
    {
        func = LogisticFunction(sigmoid);
    }
    else if(fname == "SIGMOID_PRIME")
    {
        func = LogisticFunction(sigmoid_prime);
    }
    else if(fname == "HYPERBOLIC_TANGENT")
    {
        func = LogisticFunction(hyperbolic_tangent);
    }
    else if(fname == "HYPERBOLIC_TANGENT_PRIME")
    {
        func = LogisticFunction(hyperbolic_tangent_prime);
    }

    return func;
}

}
