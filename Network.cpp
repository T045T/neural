#include "Network.h"
#include <cassert>

namespace neural {
  Network::Network(int input, int output, vector<int> hidden) :
    layerCount(hidden.size() + 1),
    inputLayer(input),
    firstHidden(),
    outputLayer()
  {
    if (hidden.size() > 0) {
      // build and link hidden layers
      firstHidden = shared_ptr<Layer>(new Layer(hidden.front(), input));
      shared_ptr<Layer> lastHidden(firstHidden);
      for (int i = 1; i < hidden.size(); i++) {
	shared_ptr<Layer> tmp(new Layer(hidden[i], lastHidden));
	lastHidden->setNextLayer(tmp);
	lastHidden = tmp;
      }
      outputLayer = shared_ptr<Layer>(new Layer(output, lastHidden));
      lastHidden->setNextLayer(outputLayer);
    } else {
      // no hidden layer!
      outputLayer = shared_ptr<Layer>(new Layer(output, input));
    }
  }
  
  Network::Network(int input, shared_ptr<Layer> hidden, shared_ptr<Layer> output) :
    layerCount(1),
    inputLayer(input),
    firstHidden(hidden),
    outputLayer(output)
  {
    if (firstHidden) {
      shared_ptr<Layer> currentLayer = firstHidden;
      while(currentLayer->nextLayer()) {
	layerCount++;
	currentLayer = currentLayer->nextLayer();
      }
      assert(currentLayer == outputLayer);
    }
  }

  Network Network::read(string &filename) {
    ifstream file(filename.c_str(), ios_base::in | ios_base::binary);
    if (!file.is_open()) {
      return Network(0,0);
    }
    Network result = read(file);
    file.close();
    return result;
  }

  Network Network::read(istream &s) {
    Network fail = Network(0,0);
    if(!s.good()) {
      return fail;
    }
    string keyword;
    s >> keyword;
    if(keyword != "NETWORK") {
      {
	return fail;
      }
    }

    s >> keyword;
    if(keyword != "input_size") {
      {
	return fail;
      }
    }
    int input_size;
    s >> input_size;
    
    s >> keyword;
    if(keyword != "layers") {
      return fail;
    }
    int layers;
    s >> layers;

    char c = s.get();
    if(c != '\n') {
      return fail;
    }

    shared_ptr<Layer> first(Layer::read(s, input_size));
    if (first->size() == 0) {
      return fail;
    }
    layers--;
    if (layers == 0) {
      // Consume trailing newline
      c = s.get();
      if (c != '\n') {
	return fail;
      }
      return Network(input_size, shared_ptr<Layer>(NULL), first);
    }
    shared_ptr<Layer> current = shared_ptr<Layer>(Layer::read(s, first));
    if (current->size() == 0) {
      return fail;
    }
    first->setNextLayer(current);
    layers--;
    while(layers > 0) {
      current->setNextLayer(shared_ptr<Layer>(Layer::read(s, current)));
      current = current->nextLayer();
      if (current->size() == 0) {
	return fail;
      }
      layers--;
    }
    // Consume trailing newline
    c = s.get();
    if (c != '\n') {
      return fail;
    }
    return Network(input_size, first, current);
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
      deltas.push_back(expected_output[i] - outputLayer->Output()[i]);
    }
    outputLayer->updateDeltas(deltas);
    startLayer->updateWeights(inputLayer, learningRate);

    // Calculate outputs with updated weights and mean squared error
    startLayer->updateOutputs(inputLayer);
    double mse = 0.0;
    for( int i = 0; i < outputLayer->size(); i++) {
      mse += (expected_output[i] - outputLayer->Output()[i]) * (expected_output[i] - outputLayer->Output()[i]);
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
    return outputLayer->Output();
  }

  bool Network::write(string &filename) const {
    // Open file in binary mode
    ofstream file(filename.c_str(), ios_base::out | ios_base::binary);
    if ( !file.is_open() ) {
      return false;
    }
    bool success = write(file);
    file.close();
    return success;
  }

  bool Network::write(ostream &s) const {
    if (!s.good() ) {
      return false;
    }
    s << "NETWORK" << "\n"
      << "input_size " << inputLayer.size() << "\n"
      << "layers " << layerCount << "\n";
    shared_ptr<Layer> currentLayer;
    if (firstHidden) {
      currentLayer = firstHidden;
    } else {
      currentLayer = outputLayer;
    }
    // Write hidden layers
    do {
      currentLayer->write(s);
      currentLayer = currentLayer->nextLayer();
    } while (currentLayer->nextLayer());
    // Write output layer
    outputLayer->write(s);
    s << endl;
    return true;
  }
}
