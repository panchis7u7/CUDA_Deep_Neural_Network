#include "Matrix.h"

// Uploaded by panchis7u7 ~ Sebastian Madrigal

Matrix::Matrix(int filas, int columnas) {
	this->filas = filas;
	this->columnas = columnas;
	this->datos = new float* [this->filas];
	for (size_t i = 0; i < this->filas; i++)
	{
		this->datos[i] = new float[this->columnas];
	}
}

Matrix::Matrix() {
	this->datos = alloc(0,0);
}

Matrix::~Matrix() {
	for (size_t i = 0; i < this->filas; i++)
	{
		delete[] datos[i];
	}
	delete[] datos;
}

float** Matrix::alloc(int filas, int columnas) {
	this->filas = filas;
	this->columnas = columnas;
	float** datos = new float* [this->filas];
	for (size_t i = 0; i < this->filas; i++)
	{
		datos[i] = new float[this->columnas];
	}
	return datos;
}

void Matrix::print() {
	for (size_t i = 0; i < filas; i++)
	{
		std::cout << "|";
		for (size_t j = 0; j < columnas; j++)
		{
			std::cout << "  " << datos[i][j] << "  ";
		}
		std::cout << "|";
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void Matrix::suma(float n) {
	for (size_t i = 0; i < this->filas; i++)
	{
		for (size_t j = 0; j < this->columnas; j++)
		{
			this->datos[i][j] += n;
		}
	}
}

void Matrix::aleatorizar() {
	for (size_t i = 0; i < this->filas; i++)
	{
		for (size_t j = 0; j < this->columnas; j++)
		{
			//Genera numero aleatorio entre -1 y 1
			this->datos[i][j] = (-1) + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (1 - (-1)))); 
		}
	}
}

void Matrix::suma(Matrix* sumando) {
	for (size_t i = 0; i < this->filas; i++)
	{
		for (size_t j = 0; j < this-> columnas; j++)
		{
			this->datos[i][j] += sumando->datos[i][j];
		}
	}
}

void Matrix::productoScalar(float n) {
	for (size_t i = 0; i < this->filas; i++)
	{
		for (size_t j = 0; j < this->columnas; j++)
		{
			this->datos[i][j] *= n;
		}
	}
}

void Matrix::productoHadamard(Matrix* A) {
	for (size_t i = 0; i < this->filas; i++)
	{
		for (size_t j = 0; j < this->columnas; j++)
		{
			this->datos[i][j] *= A->datos[i][j];
		}
	}
}

float* Matrix::toVector() {
	float* res = new float[this->filas * this->columnas];
	int k = 0;
	for (int i = 0; i < this->filas; i++)
	{
		for (int j = 0; j < this->columnas; j++)
		{
			res[k] = this->datos[i][j];
			k++;
		}
	}
	return res;
}

void Matrix::map(float (*func)(float)) {
	for (size_t i = 0; i < this->filas; i++)
	{
		for (size_t j = 0; j < this->columnas; j++)
		{
			this->datos[i][j] = func(this->datos[i][j]);
		}
	}
}