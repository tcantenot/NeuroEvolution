#ifndef NEURO_EVOLUTION_IO_UTILS_HPP
#define NEURO_EVOLUTION_IO_UTILS_HPP

#include <iosfwd>
#include <neural_network.hpp>

namespace NeuroEvolution {

template <typename T>
class Matrix;

std::ostream & operator<<(std::ostream & os, NeuralNetwork const & nn);

std::ostream & operator<<(std::ostream & os, NeuralNetwork::Network const & network);

template <typename T>
std::ostream & operator<<(std::ostream & os, Matrix<T> const & m);

}

#include "io_utils.inl"

#endif //NEURO_EVOLUTION_IO_UTILS_HPP
