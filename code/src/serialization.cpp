#include <serialization.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

#include <functions.hpp>
#include <neural_network.hpp>


namespace NeuroEvolution {

bool saveToFile(NeuralNetwork const & nn, std::string const & filename)
{
    std::ofstream f(filename);

    // Shape
    NeuralNetwork::Shape const & shape = nn.getShape();
    for(auto i = 0u; i < shape.size(); ++i)
    {
        f << shape[i] << (i != shape.size() - 1 ? " " : "");
    }
    f << std::endl;

    // Learning rate
    f << nn.getLearningRate() << std::endl;

    // Momentum
    f << nn.getMomentum() << std::endl;

    // Activation function
    f << funcToStr(nn.getActivationFunc()) << std::endl;

    // Random seed
    f << nn.getSeed() << std::endl;

    // Min value for the random weight initialization
    f << nn.getMinStartWeight() << std::endl;

    // Max value for the random weight initialization
    f << nn.getMaxStartWeight() << std::endl;

    f << std::endl;

    // Network
    NeuralNetwork::Network const & network = nn.getNetwork();
    assert(network.weights.size() == network.biases.size());
    for(auto layer = 0u; layer < network.weights.size(); ++layer)
    {
        // Weights
        auto const & w = network.weights[layer];
        for(auto i = 0u; i < w.nrows(); ++i)
        {
            for(auto j = 0u; j < w.ncols(); ++j)
            {
                f << w(i, j) << (j != w.ncols() - 1 ? " " : "");
            }
            f << std::endl;
        }

        // Biases
        auto const & b = network.biases[layer];
        for(auto i = 0u; i < b.nrows(); ++i)
        {
            assert(b.ncols() == 1);
            f << b(i, 0) << std::endl;
        }

        if(layer != network.weights.size() - 1) f << std::endl;
    }

    return true;
}

bool loadFromFile(std::string const & filename, NeuralNetwork & nn)
{
    std::ifstream f(filename);

    if(!f.is_open())
    {
        std::cerr << "Failed to open neural network file: \""
                  << filename << "\"" << std::endl;
        return false;
    }

    std::size_t n = 0u;

    std::string line;

    // Shape
    if(std::getline(f, line))
    {
        ++n;
        NeuralNetwork::Shape shape;
        std::size_t d;
        std::istringstream iss(line);
        while(iss >> d)
        {
            shape.push_back(d);
        }

        if(shape.size() < 3)
        {
            std::cerr << "Invalid shape size: " << shape.size() << std::endl;
            return false;
        }

        nn.setShape(shape);
    }
    else
    {
        std::cerr << "Cannot parse line " << n << ": "<< line << std::endl;
        return false;
    }

    // Learning rate
    if(std::getline(f, line))
    {
        std::istringstream iss(line);

        LearningRate learningRate;
        iss >> learningRate;
        nn.setLearningRate(learningRate);
    }

    // Momentum
    if(std::getline(f, line))
    {
        std::istringstream iss(line);

        Momentum momentum;
        iss >> momentum;
        nn.setMomentum(momentum);
    }

    // Activation function
    if(std::getline(f, line))
    {
        LogisticFunction func = strToFunc(line);
        nn.setActivationFunc(func);
    }

    // Random seed
    if(std::getline(f, line))
    {
        std::istringstream iss(line);

        Seed seed;
        iss >> seed;
        nn.setSeed(seed);
    }

    // Min value for the random weight initialization
    if(std::getline(f, line))
    {
        std::istringstream iss(line);

        Weight w;
        iss >> w;
        nn.setMinStartWeight(w);
    }

    // Max value for the random weight initialization
    if(std::getline(f, line))
    {
        std::istringstream iss(line);

        Weight w;
        iss >> w;
        nn.setMaxStartWeight(w);
    }

    // Empty line
    std::getline(f, line);

    // Network
    NeuralNetwork::Shape const & shape = nn.getShape();
    nn.reshape(shape);

    for(auto l = 0u; l < shape.size() - 1; ++l)
    {
        auto c = shape[l];
        auto r = shape[l+1];

        // Weights
        for(auto i = 0u; i < r; ++i)
        {
            if(std::getline(f, line))
            {
                std::istringstream iss(line);
                Weight w;
                for(auto j = 0u; j < c; ++j)
                {
                    iss >> w;
                    nn.setWeight(l, i, j, w);
                }
            }
        }

        // Biases
        for(auto i = 0u; i < r; ++i)
        {
            if(std::getline(f, line))
            {
                std::istringstream iss(line);
                Weight b;
                iss >> b;
                nn.setBias(l, i, b);
            }
        }

        // Empty line
        if(l != shape.size() - 2) std::getline(f, line);
    }

    return true;
}

}
