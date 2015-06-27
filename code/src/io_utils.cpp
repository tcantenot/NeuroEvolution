#include <io_utils.hpp>

#include <iostream>

#include <functions.hpp>
#include <matrix.hpp>


namespace NeuroEvolution {

namespace {

void printNeuralNetwork(std::ostream & os, NeuralNetwork const & nn)
{
    auto const & shape = nn.getShape();
    auto nlayers = shape.size();

    for(auto l = 0u; l < nlayers - 1; ++l)
    {
        os << "  Layer " << l<< " { " << std::endl << std::endl;

        auto I = shape[l];
        auto J = shape[l+1];

        for(auto j = 0u; j < J; ++j)
        {
            os << "    Neuron " << j << " { " << std::endl;

            for(auto i = 0u; i < I; ++i)
            {
                Weight w = nn.getWeight(l, i, j);
                os << "      Weight " << i << ": " << w << std::endl;
            }

            os << "      Bias: " << nn.getBias(l, j) << std::endl;
            os << "    }" << std::endl << std::endl;
        }

        os << "  }" << std::endl;

        if(l != nlayers - 2) os << std::endl;
    }
}

} // Anonymous namespace

std::ostream & operator<<(std::ostream & os, NeuralNetwork const & nn)
{
    os << "Neural network { " << std::endl << std::endl;

    os << "  " << "Shape:            [";
    NeuralNetwork::Shape const & shape = nn.getShape();
    for(auto i = 0u; i < shape.size(); ++i)
    {
        os << shape[i] << (i != shape.size() - 1 ? " " : "");
    }
    os << "]" << std::endl;

    os << "  " << "Learning rate:    " << nn.getLearningRate() << std::endl;
    os << "  " << "Momentum:         " << nn.getMomentum() << std::endl;
    os << "  " << "Activation func:  " << funcToStr(nn.getActivationFunc()) << std::endl;
    os << "  " << "Random seed:      " << nn.getSeed() << std::endl;
    os << "  " << "Min start weight: " << nn.getMinStartWeight() << std::endl;
    os << "  " << "Max start weight: " << nn.getMaxStartWeight() << std::endl;
    os << std::endl;
    printNeuralNetwork(os, nn);
    os << std::endl;
    os << "}";
    return os;
}

}
