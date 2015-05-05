#ifndef NEURO_EVOLUTION_IO_UTILS_HPP
#define NEURO_EVOLUTION_IO_UTILS_HPP

#include <iosfwd>

namespace NeuroEvolution {

class NeuralNetwork;

template <typename T>
class Matrix;

std::ostream & operator<<(std::ostream & os, NeuralNetwork const & nn);

template <typename T>
std::ostream & operator<<(std::ostream & os, Matrix<T> const & m);

}

#include "io_utils.inl"

#endif //NEURO_EVOLUTION_IO_UTILS_HPP
