#ifndef NEURO_EVOLUTION_MATRIX_HPP
#define NEURO_EVOLUTION_MATRIX_HPP

#include <cstdint>


namespace NeuroEvolution {

// Forward declarations

template <typename T>
class Matrix;

template <typename T>
void swap(Matrix<T> & lhs, Matrix<T> & rhs) noexcept;


/**
 * \brief Two dimensional matrix.
 */
template <typename T>
class Matrix
{
    public:
        // Construct an uninitilized MxN matrix
        Matrix(uint32_t M, uint32_t N);

        Matrix(Matrix const & other);
        Matrix(Matrix && other) noexcept;
        ~Matrix();
        Matrix<T> & operator=(Matrix other);


        // Set all the matrix coefficients to zero
        void makeZero();

        // Get the number of rows of the matrix
        uint32_t nrows() const;

        // Get the number of columns of the matrix
        uint32_t ncols() const;

        // Access the matrix coefficient at n = r * m_ncols + c
        T & operator[](uint32_t n);

        // Access the matrix coefficient at n = r * m_ncols + c
        T operator[](uint32_t n) const;

        // Access the matrix coefficient at (r, c)
        T & operator()(uint32_t r, uint32_t c);

        // Access the matrix coefficient at (r, c)
        T operator()(uint32_t r, uint32_t c) const;

        // Conversion operator to T *
        operator T *();

        // Conversion operator to T const *
        operator T const *();

        // Get the transpose of the matrix
        Matrix<T> transpose() const;

        // Compute the dot product with another matrix
        // (equivalent to a matrix post multiplication)
        Matrix<T> dot(Matrix<T> const & mat) const;

        // Component-wise addition
        Matrix<T> & operator+=(Matrix<T> const & mat);

        // Component-wise substraction
        Matrix<T> & operator-=(Matrix<T> const & mat);

        // Component wise product (Hadamard product)
        Matrix<T> & operator*=(Matrix<T> const & mat);

        // Component wise multiplication by a value
        Matrix<T> & operator*=(T v);


        // Swap two Matrix
        friend void swap<>(Matrix<T> & lhs, Matrix<T> & rhs) noexcept;


    private:
        // Construct an empty matrix
        Matrix();

    private:
        T * m;            ///< Linear array of coefficients
        uint32_t m_nrows; ///< Number of rows
        uint32_t m_ncols; ///< Number of columns
};


// Component-wise addition
template <typename T>
Matrix<T> operator+(Matrix<T> lhs, Matrix<T> const & rhs);

// Component-wise substraction
template <typename T>
Matrix<T> operator-(Matrix<T> lhs, Matrix<T> const & rhs);

// Component wise product (Hadamard product)
template <typename T>
Matrix<T> operator*(Matrix<T> lhs, Matrix<T> const & rhs);

// Component wise multiplication by a value
template <typename T>
Matrix<T> operator*(Matrix<T> mat, T v);

// Component wise multiplication by a value
template <typename T>
Matrix<T> operator*(T v, Matrix<T> mat);

}

#include "matrix.inl"

#endif //NEURO_EVOLUTION_MATRIX_HPP
