#ifndef NEURO_EVOLUTION_IO_UTILS_INL
#define NEURO_EVOLUTION_IO_UTILS_INL

#include "io_utils.hpp"

#include <iostream>

#include <matrix.hpp>

namespace NeuroEvolution {

template <typename T>
std::ostream & operator<<(std::ostream & os, Matrix<T> const & m)
{
    os << "Matrix(" << std::endl;

    for(auto r = 0u; r < m.nrows(); ++r)
    {
        for(auto c = 0u; c < m.ncols(); ++c)
        {
            os << " " << m(r, c);
        }
        os << std::endl;
    }
    os << ")";

    return os;
}

}

#endif //NEURO_EVOLUTION_IO_UTILS_INL
