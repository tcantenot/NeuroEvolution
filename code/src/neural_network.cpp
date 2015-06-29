#include <neural_network.hpp>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>

#ifndef NEURO_EVOLUTION_NN_DEBUG
#define NEURO_EVOLUTION_NN_DEBUG 0
#endif

#ifndef NEURO_EVOLUTION_NN_PROGRESSION_INFO
#define NEURO_EVOLUTION_NN_PROGRESSION_INFO 0
#endif

#ifndef NEURO_EVOLUTION_NN_NO_OUTPUT
#define NEURO_EVOLUTION_NN_NO_OUTPUT 0
#endif

#if NEURO_EVOLUTION_NN_NO_OUTPUT
#undef NEURO_EVOLUTION_NN_DEBUG
#undef NEURO_EVOLUTION_NN_PROGRESSION_INFO
#define NEURO_EVOLUTION_NN_DEBUG 0
#define NEURO_EVOLUTION_NN_PROGRESSION_INFO 0
#endif


namespace NeuroEvolution {

// Mersenne Twister random number generator
using RNG = std::mt19937;


NeuralNetwork::NeuralNetwork():
    m_shape(),
    m_learningRate(0.0),
    m_momentum(0.0),
    m_activationFunc(),
    m_activationFuncPrime(),
    m_seed(0),
    m_minStartWeight(0.0),
    m_maxStartWeight(0.0),
    m_network()
{

}

NeuralNetwork::NeuralNetwork(NeuralNetwork const & other):
    m_shape(other.m_shape),
    m_learningRate(other.m_learningRate),
    m_momentum(other.m_momentum),
    m_activationFunc(other.m_activationFunc),
    m_activationFuncPrime(other.m_activationFuncPrime),
    m_seed(other.m_seed),
    m_minStartWeight(other.m_minStartWeight),
    m_maxStartWeight(other.m_maxStartWeight),
    m_network(other.m_network)
{

}

NeuralNetwork::NeuralNetwork(NeuralNetwork && other) noexcept:
    NeuralNetwork()
{
    swap(*this, other);

}

NeuralNetwork::~NeuralNetwork()
{

}

NeuralNetwork & NeuralNetwork::operator=(NeuralNetwork other)
{
    swap(*this, other);
    return *this;
}

void swap(NeuralNetwork & lhs, NeuralNetwork & rhs) noexcept
{
    using std::swap;
    swap(lhs.m_shape, rhs.m_shape);
    swap(lhs.m_learningRate, rhs.m_learningRate);
    swap(lhs.m_momentum, rhs.m_momentum);
    swap(lhs.m_activationFunc, rhs.m_activationFunc);
    swap(lhs.m_activationFuncPrime, rhs.m_activationFuncPrime);
    swap(lhs.m_seed, rhs.m_seed);
    swap(lhs.m_minStartWeight, rhs.m_minStartWeight);
    swap(lhs.m_maxStartWeight, rhs.m_maxStartWeight);
    swap(lhs.m_network, rhs.m_network);
}


Seed NeuralNetwork::getSeed() const
{
    return m_seed;
}

NeuralNetwork::Shape NeuralNetwork::getShape() const
{
    return m_shape;
}

LearningRate NeuralNetwork::getLearningRate() const
{
    return m_learningRate;
}

Momentum NeuralNetwork::getMomentum() const
{
    return m_momentum;
}

Weight NeuralNetwork::getMinStartWeight() const
{
    return m_minStartWeight;
}

Weight NeuralNetwork::getMaxStartWeight() const
{
    return m_maxStartWeight;
}

LogisticFunction const & NeuralNetwork::getActivationFunc() const
{
    return m_activationFunc;
}

LogisticFunction const & NeuralNetwork::getActivationFuncPrime() const
{
    return m_activationFuncPrime;
}


void NeuralNetwork::setSeed(Seed seed)
{
    m_seed = seed;
}

void NeuralNetwork::setShape(NeuralNetwork::Shape const & shape)
{
    m_shape = shape;
}

void NeuralNetwork::setLearningRate(LearningRate learningRate)
{
    m_learningRate = learningRate;
}

void NeuralNetwork::setMomentum(Momentum momentum)
{
    m_momentum = momentum;
}

void NeuralNetwork::setMinStartWeight(Weight weight)
{
    m_minStartWeight = weight;
}

void NeuralNetwork::setMaxStartWeight(Weight weight)
{
    m_maxStartWeight = weight;
}

void NeuralNetwork::setActivationFunc(LogisticFunction activationFunc)
{
    m_activationFunc = activationFunc;
}

void NeuralNetwork::setActivationFuncPrime(LogisticFunction activationFuncPrime)
{
    m_activationFuncPrime = activationFuncPrime;
}



bool NeuralNetwork::synthetize()
{
    /// PREFIX CHECKS ///

    if(m_shape.size() == 0)
    {
        std::cerr << "Neural network synthesis failed: no shape" << std::endl;
        return false;
    }

    if(m_shape.size() < 2)
    {
        std::cerr << "Neural network synthesis failed: "
                  << "not enough layer (min 2 got " << m_shape.size() << ")"
                  << std::endl;
        return false;
    }

    for(auto i = 0u; i < m_shape.size(); ++i)
    {
        if(m_shape[i] == 0)
        {
            std::cerr << "Neural network synthesis failed: "
                      << "layer " << i << " has 0 neuron"
                      << std::endl;
            return false;
        }
    }

    if(!m_activationFunc)
    {
        std::cerr << "Neural network synthesis failed: "
                  << "no activation function"
                  << std::endl;

        return false;
    }

    if(!m_activationFuncPrime)
    {
        std::cerr << "Neural network synthesis failed: "
                  << "no activation prime function"
                  << std::endl;

        return false;
    }

    /// NEURAL NETWORK SYNTHESIS ///

    RNG rng;
    rng.seed(m_seed);

    std::uniform_real_distribution<Weight> realDist(m_minStartWeight, m_maxStartWeight);

    auto const random = [&realDist, &rng]()
    {
        return realDist(rng);
    };

    std::size_t previousNbNeurons = m_shape[0];

    Network network;

    // Iterate over the layers
    for(auto i = 1u; i < m_shape.size(); ++i)
    {
        // Get the number of neurons of the current layer
        std::size_t nbNeurons = m_shape[i];

        Weights weights(nbNeurons, previousNbNeurons);
        Weights biases(nbNeurons, 1);

        // Initialize randomly the weights matrix between the layers k and j
        // and the biases vector of the neurons of layer j
        for(auto j = 0u; j < nbNeurons; ++j)
        {
            for(auto k = 0u; k < previousNbNeurons; ++k)
            {
                // /!\ j and k are inverted for computation ease
                weights(j, k) = random();
            }

            biases(j, 0) = random();
        }

        previousNbNeurons = nbNeurons;

        // Add the weights matrix and the biases vector to the network
        network.weights.push_back(weights);
        network.biases.push_back(biases);
    }

    m_network = std::move(network);

    return true;
}

void NeuralNetwork::train(
    std::vector<TrainingData> trainingData,
    std::size_t epochs,
    std::size_t miniBatchSize
)
{
    this->SGD(trainingData, epochs, miniBatchSize);
}

std::vector<Weight> NeuralNetwork::compute(std::vector<Weight> const & input) const
{
    Activations activations = this->feedForward(input);
    assert(activations.ncols() == 1);

    std::vector<Weight> results(activations.nrows());
    for(auto i = 0u; i < activations.nrows(); ++i)
    {
        results[i] = activations[i];
    }

    return results;
}

NeuralNetwork::Activations NeuralNetwork::feedForward(
    std::vector<Weight> const & input,
    std::vector<Activations> * activationsList,
    std::vector<Weights> * weightedInputsList
) const
{
    // Input layer -> activations = inputs
    Activations activations(input.size(), 1);
    for(auto i = 0u; i < input.size(); ++i)
    {
        activations(i, 0) = input[i];
    }

    if(activationsList) activationsList->push_back(activations);

    // Compute the activations of all the other layers
    for(auto layer = 0u; layer < m_network.weights.size(); ++layer)
    {
        Weights const & weights = m_network.weights[layer];
        Weights const & biases  = m_network.biases[layer];

        // Weighted inputs
        Weights z = weights.dot(activations) + biases;

        if(weightedInputsList) weightedInputsList->push_back(z);

        assert(z.ncols() == 1);

        // Reshape the activations vector if necessary
        if(layer < (m_shape.size()-1) && m_shape[layer+1] != m_shape[layer])
        {
            activations = Weights(m_shape[layer+1], 1);
        }

        assert(activations.ncols() == 1);
        assert(activations.nrows() == z.nrows());

        // Compute the activations
        for(auto neuron = 0u; neuron < activations.nrows(); ++neuron)
        {
            activations[neuron] = m_activationFunc(z[neuron]);
        }

        if(activationsList) activationsList->push_back(activations);
    }

    // Return the output activations
    return activations;
}

void NeuralNetwork::SGD(
    std::vector<TrainingData> trainingData,
    std::size_t epochs,
    std::size_t miniBatchSize

)
{
    // Train the NN for a given number of epochs
    for(auto i = 0u; i < epochs; ++i)
    {
        #if !NEURO_EVOLUTION_NN_NO_OUTPUT
        std::cout << std::endl << "### Epoch "
                  << (i+1) << "/" << epochs << " ###" << std::endl;
        #endif

        // Shuffle training inputs
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(trainingData.begin(), trainingData.end(), g);

        // Split training inputs in mini-batches and train NN with each
        auto beg = trainingData.begin();
        auto end = beg;
        for(auto k = 0u; k < trainingData.size(); k += miniBatchSize)
        {
            auto m = std::min(miniBatchSize, trainingData.size() - k);
            m = std::max(m, 0ul);
            std::advance(end, m);

            #if NEURO_EVOLUTION_NN_PROGRESSION_INFO
            std::cout << "Update minibatch ["
                      << (k+1) << "-" << (k+m) << "]" << std::endl;
            #endif

            // Train the NN with the current mini-batch
            this->updateMiniBatch(beg, end, m);

            beg = end;
        }
    }
}

void NeuralNetwork::updateMiniBatch(
    TrainingDataIterator const & beg,
    TrainingDataIterator const & end,
    std::size_t m
)
{
    // Same shape as network but do not care about the content
    Nabla nabla = m_network;
    for(auto i = 0u; i < m_network.weights.size(); ++i)
    {
        nabla.weights[i].makeZero();
        nabla.biases[i].makeZero();
    }

    auto i = 0u;
    for(auto it = beg; it != end; ++it, ++i)
    {
        auto const & sample = *it;
        auto const & inputs = sample.inputs;
        auto const & expectedOutputs = sample.outputs;

        #if NEURO_EVOLUTION_NN_DEBUG
        std::cout << "Sample " << (i+1) << "/" << m << std::endl;
        #endif

        // Backpropagation
        Nabla deltaNabla = this->backPropagation(inputs, expectedOutputs);

        // Update the nablas through the network
        for(auto j = 0u; j < m_network.weights.size(); ++j)
        {
            assert(nabla.weights[j].nrows() == deltaNabla.weights[j].nrows());
            assert(nabla.weights[j].ncols() == deltaNabla.weights[j].ncols());
            assert(nabla.biases[j].nrows() == deltaNabla.biases[j].nrows());
            assert(nabla.biases[j].ncols() == deltaNabla.biases[j].ncols());
            assert(nabla.biases[j].ncols() == 1);

            nabla.weights[j] += deltaNabla.weights[j];
            nabla.biases[j]  += deltaNabla.biases[j];
        }
    }

    // Update the NN weights and biases
    auto learningRate = m_learningRate / m;
    for(auto i = 0u; i < m_network.weights.size(); ++i)
    {
        m_network.weights[i] -= learningRate * nabla.weights[i];
        m_network.biases[i]  -= learningRate * nabla.biases[i];
    }
}

NeuralNetwork::Nabla NeuralNetwork::backPropagation(
    std::vector<Weight> const & input,
    std::vector<Weight> const & expectedOutputs
)
{
    // Same shape as network but do not care about the content
    Nabla nabla = m_network;

    std::vector<Activations> activationsList;
    std::vector<Weights> weightedInputsList;

    /// FEED FORWARD ///

    this->feedForward(input, &activationsList, &weightedInputsList);

    /// BACKWARD PASS ///

    // Output layer

    auto outputActivations = activationsList[activationsList.size()-1];

    // Compute the errors between the outputs and the expected values
    Weights delta = this->costDerivative(outputActivations, expectedOutputs);

    assert(delta.ncols() == 1);

    Weights prime(delta.nrows(), 1);

    for(auto i = 0u; i < prime.nrows(); ++i)
    {
        Weight z = weightedInputsList[weightedInputsList.size()-1][i];
        prime[i] = m_activationFuncPrime(z);
    }

    delta *= prime;

    assert(delta.ncols() == 1);
    assert(nabla.weights.size() == m_network.weights.size());
    assert(nabla.biases.size() == m_network.biases.size());
    assert(nabla.weights.size() == nabla.biases.size());
    assert(nabla.weights.size() == weightedInputsList.size());
    assert(activationsList.size() == nabla.weights.size()+1);

    std::size_t last = nabla.weights.size() - 1;
    nabla.biases[last]  = delta;
    nabla.weights[last] = delta.dot(activationsList[activationsList.size()-2].transpose());

    // Other layers

    std::size_t nlayers = m_network.nlayers();

    // Backpropagate the errors through the network
    for(auto i = 2u; i < nlayers; ++i)
    {
        #if NEURO_EVOLUTION_NN_DEBUG
        std::cout << "Backward pass " << i-2 << std::endl;
        #endif

        auto idx = weightedInputsList.size() - i;

        auto const & z = weightedInputsList[idx];

        std::vector<Weight> ap;
        ap.resize(z.nrows());
        for(auto j = 0u; j < z.nrows(); ++j)
        {
            ap[j] = m_activationFuncPrime(z[j]);
        }

        auto weights = m_network.weights[idx+1].transpose();
        delta = weights.dot(delta);

        for(auto j = 0u; j < delta.nrows(); ++j)
        {
            delta[j] *= ap[j];
        }

        assert(idx == activationsList.size()-i-1);

        nabla.biases[idx]  = delta;
        nabla.weights[idx] = delta.dot(activationsList[idx].transpose());

        assert(nabla.weights[idx].nrows() > 0);
        assert(nabla.biases[idx].nrows() > 0);
        assert(nabla.biases[idx].ncols() == 1);
    }

    assert(nabla.weights.size() == nabla.biases.size());
    assert(nabla.biases.size() > 0);

    return nabla;
}

NeuralNetwork::Weights NeuralNetwork::costDerivative(
    Activations const & activations,
    std::vector<Weight> const & expectedOutputs
)
{
    assert(m_shape[m_shape.size() - 1] == expectedOutputs.size() &&
        "The number of expected outputs does not match the shape of the neural network"
    );

    assert(activations.nrows() == expectedOutputs.size());

    Weights cd(expectedOutputs.size(), 1);

    for(auto i = 0u; i < expectedOutputs.size(); ++i)
    {
        cd(i, 0) = activations(i, 0) - expectedOutputs[i];
    }

    return cd;
}


Weight NeuralNetwork::getWeight(std::size_t l, std::size_t i, std::size_t j) const
{
    return m_network.weights[l](j, i);
}

Weight NeuralNetwork::getBias(std::size_t l, std::size_t i) const
{
    return m_network.biases[l][i];
}

void NeuralNetwork::setWeight(std::size_t l, std::size_t i, std::size_t j, Weight w)
{
    m_network.weights[l](j, i) = w;
}

void NeuralNetwork::setBias(std::size_t l, std::size_t i, Weight bias)
{
    m_network.biases[l][i] = bias;
}

void NeuralNetwork::reshape(Shape shape)
{
    m_shape = shape;

    auto nlayers = shape.size();

    m_network.weights.clear();
    m_network.biases.clear();
    m_network.weights.reserve(nlayers-1);
    m_network.biases.reserve(nlayers-1);

    for(auto layer = 0u; layer < nlayers - 1; ++layer)
    {
        auto c = shape[layer];
        auto r = shape[layer+1];
        m_network.weights.push_back(Weights(r, c));
        m_network.biases.push_back(Weights(r, 1));
    }
}

}
