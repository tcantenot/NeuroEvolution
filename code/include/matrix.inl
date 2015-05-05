#ifndef NEURO_EVOLUTION_MATRIX_INL
#define NEURO_EVOLUTION_MATRIX_INL

#include "matrix.hpp"

#include <cassert>
#include <cstring>
#include <functional>

#ifndef NEURO_EVOLUTION_OMP_MATRIX
#define NEURO_EVOLUTION_OMP_MATRIX 0
#endif

#if NEURO_EVOLUTION_OMP_MATRIX
#include <omp.h>
#endif

namespace NeuroEvolution {

template <typename T>
Matrix<T>::Matrix():
    m(nullptr),
    m_nrows(0),
    m_ncols(0)
{

}

template <typename T>
Matrix<T>::Matrix(uint32_t M, uint32_t N):
    m(nullptr),
    m_nrows(M),
    m_ncols(N)
{
    m = new T[M * N];
}

template <typename T>
Matrix<T>::Matrix(Matrix const & other):
    Matrix(other.m_nrows, other.m_ncols)
{
    std::memcpy(m, other.m, m_nrows * m_ncols * sizeof(T));
}

template <typename T>
Matrix<T>::Matrix(Matrix && other) noexcept:
    Matrix()
{
    swap(*this, other);
}

template <typename T>
Matrix<T>::~Matrix()
{
    if(m) delete[] m;
}

template <typename T>
Matrix<T> & Matrix<T>::operator=(Matrix other)
{
    swap(*this, other);
    return *this;
}

template <typename T>
void Matrix<T>::makeZero()
{
    std::memset(m, 0, m_nrows * m_ncols * sizeof(T));
}

template <typename T>
uint32_t Matrix<T>::nrows() const
{
    return m_nrows;
}

template <typename T>
uint32_t Matrix<T>::ncols() const
{
    return m_ncols;
}

template <typename T>
T & Matrix<T>::operator[](uint32_t n)
{
    return m[n];
}

template <typename T>
T Matrix<T>::operator[](uint32_t n) const
{
    return m[n];
}

template <typename T>
T & Matrix<T>::operator()(uint32_t r, uint32_t c)
{
    return m[r * m_ncols + c];
}

template <typename T>
T Matrix<T>::operator()(uint32_t r, uint32_t c) const
{
    return m[r * m_ncols + c];
}

template <typename T>
Matrix<T>::operator T *()
{
    return &m[0];
}

template <typename T>
Matrix<T>::operator T const *()
{
    return &m[0];
}

template <typename T>
Matrix<T> Matrix<T>::transpose() const
{
    Matrix<T> t(m_ncols, m_nrows);

    for(auto r = 0u; r < m_nrows; ++r)
    {
        for(auto c = 0u; c < m_ncols; ++c)
        {
            t(c, r) = (*this)(r, c);
        }
    }

    return t;
}

template <typename T>
Matrix<T> Matrix<T>::dot(Matrix<T> const & mat) const
{
    assert(m_ncols == mat.m_nrows);

    Matrix<T> res(m_nrows, mat.m_ncols);

    // Use the transpose matrix to increase cache hits (better memory locality)
    Matrix<T> t = mat.transpose();

    #if NEURO_EVOLUTION_OMP_MATRIX
    #pragma omp parallel
    #endif
    {

        uint32_t i, j, k;
        #if NEURO_EVOLUTION_OMP_MATRIX
        #pragma omp for
        #endif
        for(i = 0u; i < m_nrows; ++i)
        {
            for(j = 0u; j < t.m_nrows; ++j)
            {
                T dot = 0;
                for(k = 0u; k < m_ncols; ++k)
                {
                    dot += (*this)(i, k) * t(j, k);
                }
                res(i, j) = dot;
            }
        }
    }

    return res;
}

template <typename T>
Matrix<T> & Matrix<T>::operator+=(Matrix<T> const & mat)
{
    auto n = m_nrows * m_ncols;
    for(auto i = 0u; i < n; ++i)
    {
        m[i] += mat[i];
    }

    return *this;
}

template <typename T>
Matrix<T> & Matrix<T>::operator-=(Matrix<T> const & mat)
{
    auto n = m_nrows * m_ncols;
    for(auto i = 0u; i < n; ++i)
    {
        m[i] -= mat[i];
    }

    return *this;
}

template <typename T>
Matrix<T> & Matrix<T>::operator*=(Matrix<T> const & mat)
{
    auto n = m_nrows * m_ncols;
    for(auto i = 0u; i < n; ++i)
    {
        m[i] *= mat[i];
    }

    return *this;
}

template <typename T>
Matrix<T> & Matrix<T>::operator*=(T v)
{
    auto n = m_nrows * m_ncols;
    for(auto i = 0u; i < n; ++i)
    {
        m[i] *= v;
    }

    return *this;
}

template <typename T>
void swap(Matrix<T> & lhs, Matrix<T> & rhs) noexcept
{
    using std::swap;
    swap(lhs.m, rhs.m);
    swap(lhs.m_nrows, rhs.m_nrows);
    swap(lhs.m_ncols, rhs.m_ncols);
}


template <typename T>
Matrix<T> operator+(Matrix<T> lhs, Matrix<T> const & rhs)
{
    return lhs += rhs;
}

template <typename T>
Matrix<T> operator-(Matrix<T> lhs, Matrix<T> const & rhs)
{
    return lhs -= rhs;
}

template <typename T>
Matrix<T> operator*(Matrix<T> lhs, Matrix<T> const & rhs)
{
    return lhs *= rhs;
}

template <typename T>
Matrix<T> operator*(Matrix<T> mat, T v)
{
    return mat *= v;
}

template <typename T>
Matrix<T> operator*(T v, Matrix<T> mat)
{
    return mat *= v;
}

}

#endif //NEURO_EVOLUTION_MATRIX_INL
