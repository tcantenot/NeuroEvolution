#include <io_utils.hpp>

#include <iostream>

#include <functions.hpp>
#include <matrix.hpp>


namespace NeuroEvolution {

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
    os << nn.getNetwork();
    os << std::endl;
    os << "}";
    return os;
}


std::ostream & operator<<(std::ostream & os, NeuralNetwork::Network const & network)
{
    for(auto i = 0u; i < network.weights.size(); ++i)
    {
        auto const & weights = network.weights[i];
        auto const & biases  = network.biases[i];

        os << "  Layer " << i << " { " << std::endl << std::endl;

        for(auto r = 0u; r < weights.nrows(); ++r)
        {
            os << "    Neuron " << r << " { " << std::endl;

            for(auto c = 0u; c < weights.ncols(); ++c)
            {
                Weight w = weights(r, c);
                os << "      Weight " << c << ": " << w << std::endl;
            }

            os << "      Bias: " << biases(r, 0) << std::endl;
            os << "    }" << std::endl << std::endl;
        }

        os << "  }" << std::endl;

        if(i != network.weights.size() - 1) os << std::endl;
    }

    return os;
}

}
