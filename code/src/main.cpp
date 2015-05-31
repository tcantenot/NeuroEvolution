#include <iostream>
#include <vector>

#include <functions.hpp>
#include <io_utils.hpp>
#include <matrix.hpp>
#include <neural_network_synthetizer.hpp>
#include <serialization.hpp>

using namespace NeuroEvolution;


void test_xor(NeuralNetwork & nn);
void test_or(NeuralNetwork & nn);
void test_serialization(NeuralNetwork & nn);
void benchmark_xor(NeuralNetwork && nn);

int main(int, char const **)
{
    NeuralNetworkSynthetizer nns;

    nns.setSeed(42);
    nns.setShape({2, 3, 1});
    nns.setLearningRate(1.0);
    nns.setMomentum(1.0);
    nns.setMinStartWeight(-1.0);
    nns.setMaxStartWeight(+1.0);
    nns.setActivationFunc(LogisticFunction(sigmoid));
    nns.setActivationFuncPrime(LogisticFunction(sigmoid_prime));

    //nns.setActivationFunc(LogisticFunction(hyperbolic_tangent));
    //nns.setActivationFuncPrime(LogisticFunction(hyperbolic_tangent_prime));

    NeuralNetwork nn;

    if(!nns.synthetize(nn))
    {
        std::cerr << "Failed to synthetize neural network" << std::endl;
        return -1;
    }

    std::cout << nn << std::endl;

    //test_xor(nn);
    //test_or(nn);
    //benchmark_xor(nn);
    test_serialization(nn);

    return 0;
}

void test_xor(NeuralNetwork & nn)
{
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

    nn.train(trainingInput, epochs, miniBatchSize);

    std::cout << std::endl;
    std::cout << nn << std::endl;

    std::cout << "XOR:" << std::endl;
    std::cout << "    (0, 1) = " << nn.compute({0, 1})[0] << std::endl;
    std::cout << "    (1, 0) = " << nn.compute({1, 0})[0] << std::endl;
    std::cout << "    (0, 0) = " << nn.compute({0, 0})[0] << std::endl;
    std::cout << "    (1, 1) = " << nn.compute({1, 1})[0] << std::endl;
}

void test_or(NeuralNetwork & nn)
{
    std::vector<TrainingData> trainingInput;

    for(auto i = 0u; i < 1; ++i)
    {
        trainingInput.push_back({{0, 1}, {1}});
        trainingInput.push_back({{1, 0}, {1}});
        trainingInput.push_back({{0, 0}, {0}});
        trainingInput.push_back({{1, 1}, {1}});
    }

    std::size_t epochs = 2000;
    std::size_t miniBatchSize = 1;

    nn.train(trainingInput, epochs, miniBatchSize);

    std::cout << std::endl;
    std::cout << nn << std::endl;

    std::cout << "OR:" << std::endl;
    std::cout << "    (0, 1) = " << nn.compute({0, 1})[0] << std::endl;
    std::cout << "    (1, 0) = " << nn.compute({1, 0})[0] << std::endl;
    std::cout << "    (0, 0) = " << nn.compute({0, 0})[0] << std::endl;
    std::cout << "    (1, 1) = " << nn.compute({1, 1})[0] << std::endl;
}

void benchmark_xor(NeuralNetwork && nn)
{
    NeuralNetworkSynthetizer nns;

    nns.setSeed(42);
    nns.setShape({2, 3, 1});
    nns.setLearningRate(1.0);
    nns.setMomentum(1.0);
    nns.setMinStartWeight(-1.0);
    nns.setMaxStartWeight(+1.0);
    nns.setActivationFunc(LogisticFunction(sigmoid));
    nns.setActivationFuncPrime(LogisticFunction(sigmoid_prime));


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


    std::vector<std::vector<Weight>> inputs = {{0, 1}, {1, 0}, {0, 0}, {1, 1}};
    std::vector<int32_t> expected = { 1, 1, 0, 0 };

    std::vector<Weight> avg(4, 0.0);
    uint32_t OK = 0u, total = 0u;

    auto ntests = 250u;
    for(auto i = 0u; i < ntests; ++i)
    {
        std::cout << (double(i) / ntests * 100.0) << "%" << std::endl;

        if(!nns.synthetize(nn))
        {
            std::cerr << "Failed to synthetize neural network" << std::endl;
            break;
        }

        nn.train(trainingInput, epochs, miniBatchSize);

        for(auto j = 0u; j < inputs.size(); ++j)
        {
            Weight r = nn.compute(inputs[j])[0];
            avg[j] += r;
            OK += ((r > 0.9 ? 1 : 0) == expected[j]) ? 1 : 0;
            ++total;
        }
    }

    for(auto & c : avg)
    {
        c /= static_cast<Weight>(ntests);
    }

    std::cout << "XOR:" << std::endl;
    std::cout << "    (0, 1) = " << avg[0] << std::endl;
    std::cout << "    (1, 0) = " << avg[1] << std::endl;
    std::cout << "    (0, 0) = " << avg[2] << std::endl;
    std::cout << "    (1, 1) = " << avg[3] << std::endl;

    std::cout << "Success rate = " << (double(OK) / total * 100.0) << "%"
              << " [" << OK << "/" << total << "]"
              << std::endl;
}

void test_serialization(NeuralNetwork & nn)
{
    std::string filename = "/tmp/nn.txt";
    std::cout << std::endl;
    std::cout << "Saving to file " << filename << "..." << std::endl;
    saveToFile(nn, filename);
    saveToFile(nn, "nn.txt");

    NeuralNetwork reloaded;
    std::cout << "Reloading neural network from file " << std::endl;
    if(loadFromFile(filename, reloaded))
    {
        std::cout << std::endl;
        std::cout << reloaded << std::endl;
    }
    else
    {
        std::cout << "Failed to reload nn from file " << filename << std::endl;
    }

    static auto const eq = [](Weight l, Weight r)
    {
        return std::abs(r - l) < 0.00001;
    };

    // Shape
    NeuralNetwork::Shape const & refShape = nn.getShape();
    NeuralNetwork::Shape const & shape = reloaded.getShape();
    assert(shape.size() == refShape.size());
    for(auto i = 0u; i < shape.size(); ++i)
    {
        assert(shape[i] == refShape[i]);
    }

    // Learning rate
    assert(eq(reloaded.getLearningRate(), nn.getLearningRate()));

    // Momentum
    assert(eq(reloaded.getMomentum(), nn.getMomentum()));

    // Activation function
    assert(funcToStr(reloaded.getActivationFunc()) == funcToStr(nn.getActivationFunc()));

    // Random seed
    assert(eq(reloaded.getSeed(), nn.getSeed()));

    // Min value for the random weight initialization
    assert(eq(reloaded.getMinStartWeight(), nn.getMinStartWeight()));

    // Max value for the random weight initialization
    assert(eq(reloaded.getMaxStartWeight(), nn.getMaxStartWeight()));

    // Network
    NeuralNetwork::Network const & refNetwork = nn.getNetwork();
    NeuralNetwork::Network const & network = reloaded.getNetwork();

    auto nlayers = refNetwork.nlayers();

    assert(network.nlayers() == nlayers);

    for(auto l = 0u; l < nlayers - 1; ++l)
    {
        // Weights
        auto const & refW = refNetwork.weights[l];
        auto const & w = network.weights[l];
        assert(w.nrows() == refW.nrows());
        assert(w.ncols() == refW.ncols());
        for(auto i = 0u; i < w.nrows(); ++i)
        {
            for(auto j = 0u; j < w.ncols(); ++j)
            {
                assert(eq(w(i, j), refW(i, j)));
            }
        }

        // Biases
        auto const & refB = refNetwork.biases[l];
        auto const & b = network.biases[l];
        assert(b.nrows() == refB.nrows());
        assert(b.ncols() == refB.ncols());
        assert(b.ncols() == 1);
        for(auto i = 0u; i < b.nrows(); ++i)
        {
            assert(eq(b[i], refB[i]));
        }
    }

    nn.setWeight(1, 0, 2, 42);
    nn.setWeight(0, 2, 1, 1337);
    nn.setBias(0, 1, 6);
    std::cout << std::endl;
    std::cout << nn << std::endl;
}
