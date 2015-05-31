#ifndef NEURO_EVOLUTION_SERIALIZATION_HPP
#define NEURO_EVOLUTION_SERIALIZATION_HPP

#include <string>


namespace NeuroEvolution {

class NeuralNetwork;

bool saveToFile(NeuralNetwork const & nn, std::string const & filename);
bool loadFromFile(std::string const & filename, NeuralNetwork & nn);

}

#endif //NEURO_EVOLUTION_SERIALIZATION_HPP
