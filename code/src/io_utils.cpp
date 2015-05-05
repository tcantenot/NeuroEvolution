#include <io_utils.hpp>

#include <iostream>

#include <matrix.hpp>
#include <neural_network.hpp>

namespace NeuroEvolution {

std::ostream & operator<<(std::ostream & os, NeuralNetwork const & nn)
{
    NeuralNetwork::Network const & network = nn.getNetwork();

    for(auto i = 0u; i < network.weights.size(); ++i)
    {
        auto const & weights = network.weights[i];
        auto const & biases  = network.biases[i];

        os << "Layer " << i << " { " << std::endl << std::endl;

        for(auto r = 0u; r < weights.nrows(); ++r)
        {
            os << "  Neuron " << r << " { " << std::endl;

            for(auto c = 0u; c < weights.ncols(); ++c)
            {
                Weight w = weights(r, c);
                os << "    Weight " << c << ": " << w << std::endl;
            }

            os << "    Bias: " << biases(r, 0) << std::endl;
            os << "  }" << std::endl << std::endl;
        }

        os << "}" << std::endl << std::endl;
    }

    return os;
}

}
