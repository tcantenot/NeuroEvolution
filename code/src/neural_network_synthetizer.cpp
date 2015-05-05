#include <neural_network_synthetizer.hpp>


namespace NeuroEvolution {

NeuralNetworkSynthetizer::NeuralNetworkSynthetizer():
    m_shape(),
    m_learningRate(0.0),
    m_momentum(0.0),
    m_activationFunc(),
    m_activationFuncPrime(),
    m_seed(0),
    m_minStartWeight(-1.0),
    m_maxStartWeight(+1.0)
{

}

bool NeuralNetworkSynthetizer::synthetize(NeuralNetwork & nn)
{
    NeuralNetwork tmp;
    tmp.setShape(m_shape);
    tmp.setLearningRate(m_learningRate);
    tmp.setMomentum(m_momentum);
    tmp.setActivationFunc(m_activationFunc);
    tmp.setActivationFuncPrime(m_activationFuncPrime);
    tmp.setSeed(m_seed);
    tmp.setMinStartWeight(m_minStartWeight);
    tmp.setMaxStartWeight(m_maxStartWeight);

    if(tmp.synthetize())
    {
        nn = std::move(tmp);
        return true;
    }

    return false;
}

Seed NeuralNetworkSynthetizer::getSeed() const
{
    return m_seed;
}

NeuralNetwork::Shape NeuralNetworkSynthetizer::getShape() const
{
    return m_shape;
}

LearningRate NeuralNetworkSynthetizer::getLearningRate() const
{
    return m_learningRate;
}

Momentum NeuralNetworkSynthetizer::getMomentum() const
{
    return m_momentum;
}

Weight NeuralNetworkSynthetizer::getMinStartWeight() const
{
    return m_minStartWeight;
}

Weight NeuralNetworkSynthetizer::getMaxStartWeight() const
{
    return m_maxStartWeight;
}

LogisticFunction const & NeuralNetworkSynthetizer::getActivationFunc() const
{
    return m_activationFunc;
}

LogisticFunction const & NeuralNetworkSynthetizer::getActivationFuncPrime() const
{
    return m_activationFuncPrime;
}

void NeuralNetworkSynthetizer::setSeed(Seed seed)
{
    m_seed = seed;
}

void NeuralNetworkSynthetizer::setShape(NeuralNetwork::Shape const & shape)
{
    m_shape = shape;
}

void NeuralNetworkSynthetizer::setLearningRate(LearningRate learningRate)
{
    m_learningRate = learningRate;
}

void NeuralNetworkSynthetizer::setMomentum(Momentum momentum)
{
    m_momentum = momentum;
}

void NeuralNetworkSynthetizer::setMinStartWeight(Weight weight)
{
    m_minStartWeight = weight;
}

void NeuralNetworkSynthetizer::setMaxStartWeight(Weight weight)
{
    m_maxStartWeight = weight;
}

void NeuralNetworkSynthetizer::setActivationFunc(LogisticFunction activationFunc)
{
    m_activationFunc = activationFunc;
}

void NeuralNetworkSynthetizer::setActivationFuncPrime(LogisticFunction activationFuncPrime)
{
    m_activationFuncPrime = activationFuncPrime;
}

}
