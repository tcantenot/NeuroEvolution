#include <iostream>
#include <vector>

#include <functions.hpp>
#include <io_utils.hpp>
#include <matrix.hpp>
#include <neural_network_synthetizer.hpp>

using namespace NeuroEvolution;

int main(int, char const **)
{
    NeuralNetworkSynthetizer nns;

    nns.setSeed(42);
    nns.setShape({2, 2, 1});
    nns.setLearningRate(1.0);
    nns.setMomentum(1.0);
    nns.setMinStartWeight(-1.0);
    nns.setMaxStartWeight(+1.0);
    nns.setActivationFunc(LogisticFunction(sigmoid));
    nns.setActivationFuncPrime(LogisticFunction(sigmoid_prime));

    //nns.setActivationFunc(LogisticFunction(hyperbolic_tangent));
    //nns.setActivationFuncPrime(LogisticFunction(hyperbolic_tangent_prime));

    std::vector<TrainingData> trainingInput;

    for(auto i = 0u; i < 1; ++i)
    {
        trainingInput.push_back({{0, 1}, {1}});
        trainingInput.push_back({{1, 0}, {1}});
        trainingInput.push_back({{0, 0}, {0}});
        trainingInput.push_back({{1, 1}, {0}});
    }

    std::size_t epochs = 2000;
    std::size_t miniBatchSize = 1;

    NeuralNetwork nn;

    #if 1
    if(!nns.synthetize(nn))
    {
        std::cerr << "Failed to synthetize neural network" << std::endl;
        return -1;
    }

    std::cout << nn << std::endl;

    nn.train(trainingInput, epochs, miniBatchSize);

    std::cout << std::endl;
    std::cout << nn << std::endl;

    std::cout << "(0, 1) = " << nn.compute({0, 1})[0] << std::endl;
    std::cout << "(1, 0) = " << nn.compute({1, 0})[0] << std::endl;
    std::cout << "(0, 0) = " << nn.compute({0, 0})[0] << std::endl;
    std::cout << "(1, 1) = " << nn.compute({1, 1})[0] << std::endl;
    #else
    std::vector<std::vector<Weight>> inputs = {{0, 1}, {1, 0}, {0, 0}, {1, 1}};
    std::vector<int32_t> expected = { 1, 1, 0, 0 };

    std::vector<Weight> avg(4, 0.0);
    uint32_t OK = 0u, total = 0u;

    auto ntests = 250;
    for(auto i = 0u; i < ntests; ++i)
    {
        std::cout << (double(i) / ntests * 100.0) << "%" << std::endl;

        if(!nns.synthetize(nn))
        {
            std::cerr << "Failed to synthetize neural network" << std::endl;
            return -1;
        }

        nn.SGD(trainingInput, epochs, miniBatchSize);

        for(auto j = 0u; j < inputs.size(); ++j)
        {
            Weight r = nn.feedForward(inputs[j])[0];
            avg[j] += r;
            OK += ((r > 0.9 ? 1 : 0) == expected[j]) ? 1 : 0;
            ++total;
        }
    }

    for(auto & c : avg)
    {
        c /= static_cast<Weight>(ntests);
    }

    std::cout << "(0, 1) = " << avg[0] << std::endl;
    std::cout << "(1, 0) = " << avg[1] << std::endl;
    std::cout << "(0, 0) = " << avg[2] << std::endl;
    std::cout << "(1, 1) = " << avg[3] << std::endl;

    std::cout << "Success rate = " << (double(OK) / total * 100.0) << "%"
              << " [" << OK << "/" << total << "]"
              << std::endl;
    #endif

    return 0;
}
