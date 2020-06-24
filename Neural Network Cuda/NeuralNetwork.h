#pragma once
#include "Matrix.h"

// Uploaded by panchis7u7 ~ Sebastian Madrigal

class NeuralNetwork {
public:
	NeuralNetwork(int, int, int);
	NeuralNetwork(int, std::vector<int>&, int);
	~NeuralNetwork();
	std::vector<float>* feedForward(std::vector<float>*);
	std::vector<float>* feedForwardDNN(std::vector<float>*); //DNN -> Deep Neural Network
	void train(std::vector<float>*, std::vector<float>*);
	void trainDNN(std::vector<float>*, std::vector<float>*);
	static float sigmoid(float);
	static float dsigmoid(float);
private:
	float learning_rate = 0.25f;
	int inputLayerNodes;
	int hiddenLayerSize;
	int hiddenLayerNodes;
	int outputLayerNodes;
	Matrix* pesos_ih;
	std::vector<Matrix*> pesos_hn;
	std::vector<Matrix*> bias;
	std::vector<Matrix*> salidas_capas_ocultas;
	std::vector<Matrix*> errores;
	std::vector<Matrix*> gradientes;
	std::vector<Matrix*> deltas;
	Matrix* pesos_ho;
	Matrix* bias_h;
	Matrix* bias_o;
	Matrix* salidas_capa_oculta;
protected:
};