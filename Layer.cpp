#include "Layer.h"
#include <cassert>

namespace neural {
  Layer::Layer(int neuron_count, shared_ptr<Layer> p) :
    prev(p),
    next(NULL)
  {
    if (prev) {
      input_count = prev->size();
      //prev->addNextLayer(shared_ptr<Layer>(this));
    }
    init_neurons(neuron_count, input_count);
  }
  Layer::Layer(vector<vector<double> > neuron_data, shared_ptr<Layer> p) :
    prev(p),
    next(NULL)
  {
    if (prev) {
      input_count = prev->size();
      //prev->addNextLayer(shared_ptr<Layer>(this));
    }
    init_neurons(neuron_data);
  }

  Layer::Layer(int neuron_count, int i) : 
    prev(NULL),
    next(NULL),
    input_count(i)
  {
    init_neurons(neuron_count, input_count);
  }
  Layer::Layer(vector<vector<double> > neuron_data, int i) : 
    prev(NULL),
    next(NULL),
    input_count(i)
  {
    init_neurons(neuron_data);
  }
  void Layer::addNextLayer(shared_ptr<Layer> n) {
    next = n;
  }
  void Layer::updateOutputs(vector<double> inputs) {
    output.clear();
    for (vector<Neuron>::iterator it = neurons.begin(); it != neurons.end(); it++) {
      it->updateOutput(inputs);
      output.push_back(it->Output());
    }
    if (next) {
      next->updateOutputs(this->output);
    }
  }
  void Layer::updateDeltas(vector<double> summedWeighedDeltas) {
    assert(summedWeighedDeltas.size() == this->size());
    vector<double> newDeltas(input_count);
    for (int i = 0; i < size(); i++) {
      neurons[i].updateDelta(summedWeighedDeltas[i]);
      for (int j = 0; j < newDeltas.size(); j++) {
	newDeltas[j] += neurons[i].Delta(j);
      }
    }
    if (prev) {
      prev->updateDeltas(newDeltas);
    }
  }

  void Layer::updateWeights(vector<double> inputs, double learningRate) {
    for (vector<Neuron>::iterator it = neurons.begin(); it != neurons.end(); it++) {
      it->updateWeights(inputs, learningRate);
    }
    if(next) {
      // DON'T update the output after adjusting weights, first adjust all other weights
      next->updateWeights(this->output, learningRate);
    }
  }

  void Layer::init_neurons(int neuron_count, int inputs) {
    for (int i = 0; i < neuron_count; i++) {
      // emplace_back calls the constructor inside the vector, avoiding copies
      neurons.emplace_back(inputs);
    }
  }
  void Layer::init_neurons(vector<vector<double> > neuron_data) {
    for (vector<vector<double> >::iterator it = neuron_data.begin(); it != neuron_data.end(); it++) {
      assert(it->size() == input_count);
      neurons.emplace_back(*it);
    }
  }
}
