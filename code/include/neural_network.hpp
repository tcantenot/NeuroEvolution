#ifndef NEURO_EVOLUTION_NEURAL_NETWORK_HPP
#define NEURO_EVOLUTION_NEURAL_NETWORK_HPP

#include <cstdint>
#include <iosfwd>
#include <vector>

#include <matrix.hpp>
#include <types.hpp>


namespace NeuroEvolution {

/**
 * \brief Class representing a neural network.
 *
 * The neural network must first be created and configurated.
 * Various parameters can be set like its:
 *  - shape: list of dimensions, one per layer (first layer = input layer)
 *  - learning rate
 *  - activation function
 *  - etc.
 * (See also NeuralNetworkSynthetizer which allows to build several NNs with
 * more or less the same parameters more easily).
 *
 * After configuration, the neural network must be synthetized (initialized)
 * with the method NeuralNetwork::synthetize.
 *
 * After initialization, it must be trained via the method NeuralNetwork::train.
 *
 * After the training, the neural network is ready to use via the method
 * NeuralNetwork::compute.
 *
 */
class NeuralNetwork
{
    public:
        //! Neural network shape type
        using Shape = std::vector<std::size_t>;

    public:
        NeuralNetwork();
        NeuralNetwork(NeuralNetwork const & other);
        NeuralNetwork(NeuralNetwork && other) noexcept;
        ~NeuralNetwork();

        NeuralNetwork & operator=(NeuralNetwork other);

        friend void swap(NeuralNetwork & lhs, NeuralNetwork & rhs) noexcept;


    // Getters/Setters
    public:
        Seed getSeed() const;
        Shape getShape() const;
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


    public:
        // Initialization
        bool synthetize();

        // Training
        void train(
            std::vector<TrainingData> trainingData,
            std::size_t epochs,
            std::size_t miniBatchSize
        );

        // Compute
        std::vector<Weight> compute(std::vector<Weight> const & input) const;


    public:

        //! Weight matrix type
        using Weights = Matrix<Weight>;


        /** Internal neural network */
        struct Network
        {
            Network(): weights(), biases() { }

            std::size_t nlayers() const
            {
                return weights.size() + 1;
            }

            //! Weights (list of weight matrices)
            // #(weight matrices) = nlayers - 1 (one between each subsequent layers)
            std::vector<Weights> weights;

            //! Biases (list of biases vectors)
            // #(biases vector) = nlayers - 1 (none for input layer)
            std::vector<Weights> biases;
        };


        // Get the internal network of the NN
        Network const & getNetwork() const;

        // Get the weight between neuron j in the layer l+1 and neuron i in layer l
        Weight getWeight(std::size_t l, std::size_t i, std::size_t j) const;

        // Get the bias of neuron i in the layer l
        Weight getBias(std::size_t l, std::size_t i) const;

        // Set the weight between neuron j in the layer l+1 and neuron i in layer l
        void setWeight(std::size_t l, std::size_t i, std::size_t j, Weight w);

        // Set the bias of neuron i in the layer l
        void setBias(std::size_t l, std::size_t i, Weight bias);

        // Reshape the neural network
        void reshape(Shape shape);


    private:
        //! Activation vector type (matrix with one column <-> vector)
        using Activations = Weights;

        // Feed forward pass
        Activations feedForward(
            std::vector<Weight> const & inputData,
            std::vector<Activations> * activationsList = nullptr,
            std::vector<Weights> * weightedInputsList = nullptr
        ) const;

        // Stochastic gradient descent
        void SGD(
            std::vector<TrainingData> trainingData,
            std::size_t epochs,
            std::size_t miniBatchSize
        );


        using TrainingDataIterator = std::vector<TrainingData>::iterator;

        // Update NN with a random mini batch of training data
        void updateMiniBatch(
            TrainingDataIterator const & beg,
            TrainingDataIterator const & end,
            std::size_t m
        );


        using Nabla = Network;

        // Back propagation
        Nabla backPropagation(
            std::vector<Weight> const & input,
            std::vector<Weight> const & expectedOutput
        );

        // Cost function derivative
        Weights costDerivative(
            Activations const & activations,
            std::vector<Weight> const & expectedOutput
        );


    private:
        //! NN shape
        Shape m_shape;

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

        //! Internal neural network
        Network m_network;
};

}

#endif //NEURO_EVOLUTION_NEURAL_NETWORK_HPP
