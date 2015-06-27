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

    // Activation function prime
    f << funcToStr(nn.getActivationFuncPrime()) << std::endl;

    // Random seed
    f << nn.getSeed() << std::endl;

    // Min value for the random weight initialization
    f << nn.getMinStartWeight() << std::endl;

    // Max value for the random weight initialization
    f << nn.getMaxStartWeight() << std::endl;

    f << std::endl;

    // Network
    auto const & nnshape = nn.getShape();
    auto nlayers = nnshape.size();
    for(auto l = 0u; l < nlayers-1; ++l)
    {
        auto I = nnshape[l];
        auto J = nnshape[l+1];

        // Weights
        for(auto j = 0u; j < J; ++j)
        {
            for(auto i = 0u; i < I; ++i)
            {
                f << nn.getWeight(l, i, j) << (i != I - 1 ? " " : "");
            }
            f << std::endl;
        }

        f << std::endl;

        // Biases
        for(auto j = 0u; j < J; ++j)
        {
            f << nn.getBias(l, j) << std::endl;
        }

        if(l != nlayers - 2) f << std::endl << std::endl;
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

    // Activation function prime
    if(std::getline(f, line))
    {
        LogisticFunction funcPrime = strToFunc(line);
        nn.setActivationFuncPrime(funcPrime);
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

    auto const & nnshape = nn.getShape();
    auto nlayers = nnshape.size();
    for(auto l = 0u; l < nlayers-1; ++l)
    {
        auto I = nnshape[l];
        auto J = nnshape[l+1];

        // Weights
        for(auto j = 0u; j < J; ++j)
        {
            if(std::getline(f, line))
            {
                std::istringstream iss(line);
                Weight w;
                for(auto i = 0u; i < I; ++i)
                {
                    iss >> w;
                    nn.setWeight(l, i, j, w);
                }
            }
        }

        // Empty line
        std::getline(f, line);

        // Biases
        for(auto j = 0u; j < J; ++j)
        {
            if(std::getline(f, line))
            {
                std::istringstream iss(line);
                Weight b;
                iss >> b;
                nn.setBias(l, j, b);
            }
        }

        // Empty line
        std::getline(f, line);

        // Empty line
        if(l != nlayers - 2) std::getline(f, line);
    }

    return true;
}

}
