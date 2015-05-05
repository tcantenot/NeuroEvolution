#ifndef NEURO_EVOLUTION_TYPES_HPP
#define NEURO_EVOLUTION_TYPES_HPP

#include <cstdint>
#include <functional>


namespace NeuroEvolution {

//! Random seed type
using Seed = uint32_t;

//! Neural network weight type
using Weight = double;

//! Neural network learning rate type
using LearningRate = Weight;

//! Neural network momentum type
using Momentum = Weight;

//! Neural network logistic function type
using LogisticFunction = std::function<Weight(Weight)>;


/**
 * \brief Class representing training data.
 * Contains a list of inputs with the corresponding list of expected outputs
 */
struct TrainingData
{
    using Data = std::vector<Weight>;

    Data inputs;  ///< Inputs
    Data outputs; ///< Expected outputs

    TrainingData(): inputs(), outputs()
    {

    }

    TrainingData(Data const & in, Data const & out):
        inputs(in), outputs(out)
    {

    }
};

}

#endif //NEURO_EVOLUTION_TYPES_HPP
