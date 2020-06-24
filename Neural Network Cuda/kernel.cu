#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Matrix.h"
#include "NeuralNetwork.h"
#include <vector>
#include <stdio.h>

std::vector<Matrix*> Matrices;
std::vector<float*> devicePtrs;

__global__ void matrixMul(int* A, int* B, int* C,
	int aF, int aC,
	int bF, int bC,
	int cF, int cC) {
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (aC != bF) return;
	if ((row < aF) && (col < bC)) {
		for (int k = 0; k < aC; ++k) {
			C[row * cC + col] += A[row * aC + k] * B[k * bC + col];
		}
	}
}

void prepareCuda() {
	devicePtrs.reserve(Matrices.size());
	for (size_t i = 0; i < Matrices.size(); i++)
	{
		devicePtrs.push_back(new float);
		if (cudaMalloc((void**)&devicePtrs[i], sizeof(float) * Matrices[i]->columnas * Matrices[i]->filas) != cudaSuccess)
			std::cout << "Error al despachar memoria a la matriz: " << i << "." << std::endl;
		// Copy data to the device
		if (cudaMemcpy(devicePtrs[i], Matrices[i]->toVector(), sizeof(float) * Matrices[i]->columnas * Matrices[i]->filas, cudaMemcpyHostToDevice) != cudaSuccess)
			std::cout << "Error en MemCpy con la matriz: " << i << "." << std::endl;
	}
}

NeuralNetwork::NeuralNetwork(int i, int h, int o) {
	this->inputLayerNodes = i;
	this->hiddenLayerNodes = h;
	this->hiddenLayerSize = 1;
	this->outputLayerNodes = o;
	//Matriz que representa los pesos entre las capa de Entrada-Oculta
	this->pesos_ih = new Matrix(this->hiddenLayerNodes, this->inputLayerNodes);
	//Matriz que representa los pesos entre las capa Oculta-Salida
	this->pesos_ho = new Matrix(this->outputLayerNodes, this->hiddenLayerNodes);
	//Se asigna valores aleatorios a las matrices
	this->pesos_ih->aleatorizar();
	this->pesos_ho->aleatorizar();
	//Se asigna un sesgo o predisposicion a las neuronas
	this->bias_h = new Matrix(this->hiddenLayerNodes, 1);
	this->bias_o = new Matrix(this->outputLayerNodes, 1);
	//Se asigna valores aleatorios a las matrices
	this->bias_h->aleatorizar();
	this->bias_o->aleatorizar();
	this->salidas_capa_oculta = nullptr;
}

NeuralNetwork::NeuralNetwork(int i, std::vector<int>& h, int o) {
	this->inputLayerNodes = i;
	this->hiddenLayerSize = h.size();
	this->outputLayerNodes = o;
	//Matriz que representa los pesos entre las capa de Entrada-Oculta[0]
	this->pesos_ih = new Matrix(h[0], this->inputLayerNodes);
	//Se asigna valores aleatorios a las matrices
	this->pesos_ih->aleatorizar();
	//reserva espacio en memoria para los pesos de las n capas ocultas de la red (Optimizacion, evita la redimiension del vector cada vez que se inserta).
	Matrices.reserve(h.size() + 2);
	this->pesos_hn.reserve(h.size() - 1);
	this->gradientes.reserve(h.size() - 1);
	this->deltas.reserve(h.size() - 1);
	this->salidas_capas_ocultas.reserve(h.size());
	this->errores.reserve(h.size());
	this->bias.reserve(h.size() + 1);
	Matrices.push_back(pesos_ih);

	for (int i = 0; i < h.size(); i++)
	{
		errores.push_back(nullptr);
		salidas_capas_ocultas.push_back(nullptr);
		this->bias.push_back(new Matrix(h[i], 1));
		this->bias.at(i)->aleatorizar();
	}
	//reserva espacio en memoria para los sesgos de las n capas ocultas y de salida de la red (Optimizacion, evita la redimiension del vector cada vez que se inserta).

	for (size_t i = 0; i < h.size() - 1; i++)
	{
		this->gradientes.push_back(nullptr);
		this->deltas.push_back(nullptr);
		this->pesos_hn.push_back(new Matrix(h[i + 1], h[i]));
		this->pesos_hn.at(i)->aleatorizar();
		Matrices.push_back(pesos_hn[i]);
	}
	//Matriz que representa los pesos entre las capa enesima(oculta)-Salida y se aleatoriza
	this->pesos_ho = new Matrix(this->outputLayerNodes, h[h.size() - 1]);
	this->pesos_ho->aleatorizar();
	Matrices.push_back(pesos_ho);
	//Se asigna un sesgo o predisposicion a la enesima capa oculta y se aleatoriza.
	//Matriz que representa el sesgo de la capa oculta y se aleatoriza
	this->bias.push_back(new Matrix(outputLayerNodes, 1));
	this->bias.at(bias.size() - 1)->aleatorizar();
	//Variables por eliminar
	this->hiddenLayerNodes = 0;
	this->salidas_capa_oculta = nullptr;
	this->bias_h = nullptr;
	this->bias_o = nullptr;
	prepareCuda();
}

NeuralNetwork::~NeuralNetwork() {
	delete(pesos_ih);
	delete(pesos_ho);
	for (auto& matrix : this->errores) {
		delete matrix;
	}
	for (auto& matrix : this->gradientes) {
		delete matrix;
	}
	for (auto& matrix : this->deltas) {
		delete matrix;
	}
	for (auto& matrix : this->salidas_capas_ocultas) {
		delete matrix;
	}
}

std::vector<float>* NeuralNetwork::feedForwardDNN(std::vector<float>* vec_entrada) {
	//----Generando las salidas de la capas ocultas----
	//Convierte el vector de entrada a una matriz
	Matrix* entradas = Matrix::fromVector(vec_entrada);
	//Se multiplica la matriz de pesos entre la capa oculta y la matriz de entradas
	Matrix* entradas_capa_oculta = Matrix::multiplicar(this->pesos_ih, entradas);
	//Al resultado de la multiplicacion se le agrega el sesgo
	entradas_capa_oculta->suma(this->bias.at(0));
	//sig((W * i) + b) se aplica la funcion sigmoide
	////this->salidas_capas_ocultas.push_back(Matrix::map(entradas_capa_oculta, sigmoid));

	this->salidas_capas_ocultas.at(0) = Matrix::map(entradas_capa_oculta, sigmoid);
	//Se multiplica la matriz de pesos entre la capas ocultas y la matriz de entradas previas
	for (size_t i = 0; i < (hiddenLayerSize - 1); i++)
	{
		//Se multiplica la matriz de pesos entre la capa oculta y la matriz de entradas de la enesima capa oculta
		////this->salidas_capas_ocultas.push_back(Matrix::multiplicar(this->pesos_hn.at(i), this->salidas_capas_ocultas.at(i)));
		this->salidas_capas_ocultas.at(i + 1) = Matrix::multiplicar(this->pesos_hn.at(i), this->salidas_capas_ocultas.at(i));
		//Al resultado de la multiplicacion se le agrega el sesgo
		this->salidas_capas_ocultas.at(i + 1)->suma(this->bias.at(i + 1));
		//sig((W * i) + b) se aplica la funcion sigmoide
		this->salidas_capas_ocultas.at(i + 1)->map(sigmoid);
	}
	//----Generando las salida----
	//Se multiplica la matriz de pesos entre la capa de salida y la matriz de salidas de la capa oculta
	Matrix* entradas_capa_salida = Matrix::multiplicar(this->pesos_ho, this->salidas_capas_ocultas.at(salidas_capas_ocultas.size() - 1));
	//Al resultado de la multiplicacion se le agrega el sesgo
	entradas_capa_salida->suma(bias.at(hiddenLayerSize));
	//sig((W * i) * b) se aplica la funcion sigmoide
	Matrix* salidas = Matrix::map(entradas_capa_salida, sigmoid);
	//delete entradas_capa_oculta;
	return Matrix::toVector(salidas);
}

std::vector<float>* NeuralNetwork::feedForward(std::vector<float>* vec_entrada) {
	/*
		Generando las salidas de la capa oculta
	*/
	//Convierte el vector de entrada a una matriz
	Matrix* entradas = Matrix::fromVector(vec_entrada);
	//Se multiplica la matriz de pesos entre la capa oculta y la matriz de entradas
	Matrix* entradas_capa_oculta = Matrix::multiplicar(this->pesos_ih, entradas);
	//Al resultado de la multiplicacion se le agrega el sesgo
	entradas_capa_oculta->suma(this->bias_h);
	//sig((W * i) + b) se aplica la funcion sigmoide
	this->salidas_capa_oculta = Matrix::map(entradas_capa_oculta, sigmoid);
	/*
		Generando la salida
	*/
	//Se multiplica la matriz de pesos entre la capa de salida y la matriz de salidas de la capa oculta
	Matrix* entradas_capa_salida = Matrix::multiplicar(this->pesos_ho, this->salidas_capa_oculta);
	//Al resultado de la multiplicacion se le agrega el sesgo
	entradas_capa_salida->suma(this->bias_o);
	//sig((W * i) * b) se aplica la funcion sigmoide
	Matrix* salidas = Matrix::map(entradas_capa_salida, sigmoid);
	return Matrix::toVector(salidas);
}

void NeuralNetwork::train(std::vector<float>* vec_entradas, std::vector<float>* vec_respuestas) {
	std::vector<float>* vec_salidas = this->feedForward(vec_entradas);
	//Convertir vectores a matrices
	Matrix* entradas = Matrix::fromVector(vec_entradas);
	Matrix* respuestas = Matrix::fromVector(vec_respuestas);
	Matrix* salidas = Matrix::fromVector(vec_salidas);

	//Calcular el error => respuestas - salidas
	Matrix* errores_salida = Matrix::restaElementWise(respuestas, salidas);

	//Calcular los errores de la capa oculta->salida 
	Matrix* errores_capa_oculta_salida = Matrix::multiplicar(Matrix::transpuesta(pesos_ho), errores_salida);

	//Calcular el gradiente de la capa de salida = learning_rate * errores_salida * dsigmoid(salidas)
	Matrix* gradiente_salida = Matrix::map(salidas, dsigmoid);
	gradiente_salida->productoHadamard(errores_salida);
	gradiente_salida->productoScalar(learning_rate);

	//Calcular los gradientes de la capa oculta = learning_rate * errores_capa_oculta * dsigmoid(salidas_capa_oculta)
	Matrix* gradientes_capa_oculta = Matrix::map(this->salidas_capa_oculta, dsigmoid);
	gradientes_capa_oculta->productoHadamard(errores_capa_oculta_salida);
	gradientes_capa_oculta->productoScalar(learning_rate);

	//Calcular deltas de la capa oculta-salida
	//pesos_delta = learning_rate * errores * dsigmoid(salidas) * pesos(T)
	Matrix* deltas_pesos_ho = Matrix::multiplicar(gradiente_salida, Matrix::transpuesta(this->salidas_capa_oculta));
	this->pesos_ho->suma(deltas_pesos_ho);
	this->bias_o->suma(gradiente_salida);

	//Calcular deltas de la capa de entrada-oculta
	Matrix* deltas_pesos_ih = Matrix::multiplicar(gradientes_capa_oculta, Matrix::transpuesta(entradas));
	this->pesos_ih->suma(deltas_pesos_ih);
	this->bias_h->suma(gradientes_capa_oculta);
}

void NeuralNetwork::trainDNN(std::vector<float>* vec_entradas, std::vector<float>* vec_respuestas) {
	std::vector<float>* vec_salidas = this->feedForwardDNN(vec_entradas);
	//Convertir vectores a matrices
	Matrix* entradas = Matrix::fromVector(vec_entradas);
	Matrix* respuestas = Matrix::fromVector(vec_respuestas);
	Matrix* salidas = Matrix::fromVector(vec_salidas);

	//Calcular el error => respuestas - salidas
	Matrix* errores_salida = Matrix::restaElementWise(respuestas, salidas);

	//Calcular los errores de la capa oculta->salida 

	this->errores.at(hiddenLayerSize - 1) = Matrix::multiplicar(Matrix::transpuesta(this->pesos_ho), errores_salida);
	for (int i = hiddenLayerSize - 1; i > 0; i--)
	{
		this->errores.at(i - 1) = Matrix::multiplicar(Matrix::transpuesta(this->pesos_hn.at(i - 1)), errores.at(i));
	}

	//Calcular el gradiente de la capa de salida = learning_rate * errores_salida * dsigmoid(salidas)
	Matrix* gradiente_salida = Matrix::map(salidas, dsigmoid);
	gradiente_salida->productoHadamard(errores_salida);
	gradiente_salida->productoScalar(learning_rate);
	this->bias.at(bias.size() - 1)->suma(gradiente_salida);
	//Calcular el gradiente de las capas ocultas y oculta-salida = learning_rate * errores_salida * dsigmoid(salidas)
	for (int i = hiddenLayerSize - 1; i > 0; i--)
	{
		this->gradientes.at(i - 1) = Matrix::map(this->salidas_capas_ocultas.at(hiddenLayerSize - i), dsigmoid);
		this->gradientes.at(i - 1)->productoHadamard(errores.at(hiddenLayerSize - i));
		this->gradientes.at(i - 1)->productoScalar(learning_rate);
		this->bias.at(hiddenLayerSize - i)->suma(gradientes.at(i - 1));
	}
	//Calcular los gradientes de la capa entada-oculta = learning_rate * errores_capa_oculta * dsigmoid(salidas_capa_oculta)
	Matrix* gradiente_entrada_oculta = Matrix::map(this->salidas_capas_ocultas.at(0), dsigmoid);
	gradiente_entrada_oculta->productoHadamard(errores.at(0));
	gradiente_entrada_oculta->productoScalar(learning_rate);
	this->bias.at(0)->suma(gradiente_entrada_oculta);

	//Calcular deltas de la capa oculta-salida
	Matrix* deltas_pesos_ho = Matrix::multiplicar(gradiente_salida, Matrix::transpuesta(this->salidas_capas_ocultas.at(salidas_capas_ocultas.size() - 1)));
	this->pesos_ho->suma(deltas_pesos_ho);
	for (int i = hiddenLayerSize - 1; i > 0; i--)
	{
		//this->deltas.at(i-1) = Matrix::multiplicar(gradientes.at(i-1), Matrix::transpuesta(this->salidas_capas_ocultas.at(i)));
		this->deltas.at(i - 1) = Matrix::multiplicar(this->salidas_capas_ocultas.at(i), Matrix::transpuesta(gradientes.at(i - 1)));
		this->pesos_hn.at(i - 1)->suma(deltas.at(i - 1));
	}
	//Calcular deltas de la capa de entrada-oculta
	Matrix* deltas_pesos_ih = Matrix::multiplicar(gradiente_entrada_oculta, Matrix::transpuesta(entradas));
	this->pesos_ih->suma(deltas_pesos_ih);
}

float NeuralNetwork::sigmoid(float n) {
	return (1 / (1 + pow(2.718281828, (-n))));
}

float NeuralNetwork::dsigmoid(float y) {
	//return sigmoid(n) * (1 - sigmoid(n));
	return (y * (1 - y));
}

int main() {
	srand(static_cast<unsigned>(time(0)));
	NeuralNetwork* nn = new NeuralNetwork(2, 4, 1);
	std::vector<float> entradas[] = { { 0 , 0},
									  { 1 , 0},
									  { 0 , 1},
									  { 1 , 1} };
	std::vector<float> esperado[] = { {0},{1},{1},{0} };
	for (size_t i = 0; i < 30000; i++)
	{
		int index = rand() % 4;
		nn->train(&entradas[index], &esperado[index]);
	}
	std::cout << "0,0: " << nn->feedForward(&entradas[0])->at(0) << std::endl;
	std::cout << "0,1: " << nn->feedForward(&entradas[1])->at(0) << std::endl;
	std::cout << "1,0: " << nn->feedForward(&entradas[2])->at(0) << std::endl;
	std::cout << "1,1: " << nn->feedForward(&entradas[3])->at(0) << std::endl;

	std::cout << std::endl;
	std::vector<int> f1 = { 4 };
	NeuralNetwork* nn2 = new NeuralNetwork(2, f1, 1);
	for (size_t i = 0; i < 15000; i++)
	{
		int index = rand() % 4;
		nn2->trainDNN(&entradas[index], &esperado[index]);
	}
	std::cout << "0,0: " << nn2->feedForwardDNN(&entradas[0])->at(0) << std::endl;
	std::cout << "0,1: " << nn2->feedForwardDNN(&entradas[1])->at(0) << std::endl;
	std::cout << "1,0: " << nn2->feedForwardDNN(&entradas[2])->at(0) << std::endl;
	std::cout << "1,1: " << nn2->feedForwardDNN(&entradas[3])->at(0) << std::endl;
	return 0;
}

//cudaError_t cudaStatus;
//// Check for any errors launching the kernel
//cudaStatus = cudaGetLastError();
//if (cudaStatus != cudaSuccess) {
//    fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//    goto Error;
//}
//
//// cudaDeviceSynchronize waits for the kernel to finish, and returns
//// any errors encountered during the launch.
//cudaStatus = cudaDeviceSynchronize();
//if (cudaStatus != cudaSuccess) {
//    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//    goto Error;
//}