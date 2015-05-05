#ifndef NEURAL_EVOLUTION_NEURAL_NETWORK_SYNTHETIZER_HPP
#define NEURAL_EVOLUTION_NEURAL_NETWORK_SYNTHETIZER_HPP

#include <neural_network.hpp>
#include <types.hpp>


namespace NeuroEvolution {

/**
 * Neural Network Synthetizer (builder pattern).
 */
class NeuralNetworkSynthetizer
{
    public:
        NeuralNetworkSynthetizer();

        /** Synthethize a neural network */
        bool synthetize(NeuralNetwork & nn);


    // Getters/Setters
    public:
        Seed getSeed() const;
        NeuralNetwork::Shape getShape() const;
        LearningRate getLearningRate() const;
        Momentum getMomentum() const;
        Weight getMinStartWeight() const;
        Weight getMaxStartWeight() const;
        LogisticFunction const & getActivationFunc() const;
        LogisticFunction const & getActivationFuncPrime() const;

        void setSeed(Seed seed);
        void setShape(NeuralNetwork::Shape const & shape);
        void setLearningRate(LearningRate learningRate);
        void setMomentum(Momentum momentum);
        void setMinStartWeight(Weight weight);
        void setMaxStartWeight(Weight weight);
        void setActivationFunc(LogisticFunction activationFunc);
        void setActivationFuncPrime(LogisticFunction activationFuncPrime);


    private:
        //! NN shape
        NeuralNetwork::Shape m_shape;

        //! NN learning rate
        LearningRate m_learningRate;

        //! NN momentum
        Momentum m_momentum;

        //! Activation function
        LogisticFunction m_activationFunc;

        //! Activation function derivative
        LogisticFunction m_activationFuncPrime;

        //! Random seed
        Seed m_seed;

        //! Min value for the random weight initialization
        Weight m_minStartWeight;

        //! Max value for the random weight initialization
        Weight m_maxStartWeight;
};

}

#endif //NEURAL_EVOLUTION_NEURAL_NETWORK_SYNTHETIZER_HPP
