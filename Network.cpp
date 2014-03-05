#include "Network.h"
#include <cassert>

namespace neural {
  Network::Network(int input, int output, vector<int> hidden) :
    inputLayer(input),
    firstHidden(),
    outputLayer()
  {
    assert(input > 0);
    assert(output > 0);
    if (hidden.size() > 0) {
      // build and link hidden layers
      firstHidden = shared_ptr<Layer>(new Layer(hidden.front(), input));
      shared_ptr<Layer> lastHidden(firstHidden);
      for (int i = 1; i < hidden.size(); i++) {
	shared_ptr<Layer> tmp(new Layer(hidden[i], lastHidden));
	lastHidden->addNextLayer(tmp);
	lastHidden = tmp;
      }
      outputLayer = shared_ptr<Layer>(new Layer(output, lastHidden));
      lastHidden->addNextLayer(outputLayer);
    } else {
      // no hidden layer!
      outputLayer = shared_ptr<Layer>(new Layer(output, input));
    }
  }

  double Network::trainSingle(vector<double> input, vector<double> expected_output, double learningRate) {
    assert(input.size() == inputLayer.size());
    assert(expected_output.size() == outputLayer->size());
    inputLayer = input;
    shared_ptr<Layer> startLayer;
    if (firstHidden) {
      startLayer = firstHidden;
    } else {
      startLayer = outputLayer;
    }

    startLayer->updateOutputs(inputLayer);
    vector<double> deltas;
    for( int i = 0; i < outputLayer->size(); i++) {
      deltas.push_back(expected_output[i] - outputLayer->output[i]);
    }
    outputLayer->updateDeltas(deltas);
    startLayer->updateWeights(inputLayer, learningRate);

    // Calculate outputs with updated weights and mean squared error
    startLayer->updateOutputs(inputLayer);
    double mse = 0.0;
    for( int i = 0; i < outputLayer->size(); i++) {
      mse += (expected_output[i] - outputLayer->output[i]) * (expected_output[i] - outputLayer->output[i]);
    }
    return mse / (double) outputLayer->size();
  }

  vector<double> Network::run(vector<double> input) {
    shared_ptr<Layer> startLayer;
    if (firstHidden) {
      startLayer = firstHidden;
    } else {
      startLayer = outputLayer;
    }
    startLayer->updateOutputs(input);
    return outputLayer->output;
  }
}
