#include "neural/Layer.h"
#include <cassert>

namespace neural {
  Layer::Layer(int neuron_count, shared_ptr<Layer> previous) :
    prev(previous),
    next()
  {
    if (prev) {
      input_count = prev->size();
    }
    init_neurons(neuron_count, input_count);
  }
  Layer::Layer(vector<vector<double> > neuron_data, shared_ptr<Layer> previous) :
    prev(previous),
    next()
  {
    if (prev) {
      input_count = prev->size();
    }
    init_neurons(neuron_data);
  }

  Layer::Layer(int neuron_count, int inputs) : 
    prev(NULL),
    next(NULL),
    input_count(inputs)
  {
    init_neurons(neuron_count, input_count);
  }
  Layer::Layer(vector<vector<double> > neuron_data, int inputs) : 
    prev(NULL),
    next(NULL),
    input_count(inputs)
  {
    init_neurons(neuron_data);
  }
  Layer::Layer(vector<Neuron> neuron_vector, shared_ptr<Layer> previous) :
    prev(previous),
    next(NULL),
    input_count(previous->size()),
    neurons(neuron_vector)
  {}
  Layer::Layer(vector<Neuron> neuron_vector, int inputs) :
    prev(NULL),
    next(NULL),
    input_count(inputs),
    neurons(neuron_vector)
  {}

  /**
   * Read a serialized Layer from a file
   * @param s an input stream containing a serialized Layer
   * @param p the previous(ly deserialized) Layer
   * @return a Layer equivalent to what was serialized if reading was successful, an empty Layer (0 neurons, 0 inputs) if it wasn't
   */
  Layer* Layer::read(istream &s, shared_ptr<Layer> previous) {
    Layer* fail = new Layer(0,0);
    if(!s.good()) return fail;
    std::string keyword;
    s >> keyword;
    if(keyword != "LAYER") return fail;
    
    s >> keyword;
    if(keyword != "inputs") return fail;
    int inputs;
    s >> inputs;
    if(inputs != previous->size()) return fail;
    
    s >> keyword;
    if(keyword != "neurons") return fail;
    int neuron_count;
    s >> neuron_count;
    // consume newline
    char c = s.get();
    if (c != '\n') return fail;

    vector<Neuron> neuron_vector = readNeurons(s, neuron_count);
    if (neuron_vector.size() == 0) return fail;
    delete fail;
    return new Layer(neuron_vector, previous);
  }

  Layer* Layer::read(istream &s, int input_size) {
    Layer* fail = new Layer(0,0);
    if(!s.good()) return fail;
    std::string keyword;
    s >> keyword;
    if(keyword != "LAYER") {
      cerr << "Missing LAYER keyword";
      return fail;
    }
    
    s >> keyword;
    if(keyword != "inputs") {
      cerr << "Missing inputs" << endl;
      return fail;
    }
    int inputs;
    s >> inputs;
    if(inputs != input_size) {
      cerr << "Wrong input size!" << endl;
      return fail;
    }
    
    s >> keyword;
    if(keyword != "neurons") {
      cerr << "Missing neuron count" << endl;
      return fail;
    }
    int neuron_count;
    s >> neuron_count;
    // consume newline
    char c = s.get();
    if (c != '\n') {
      cerr << "Missing newline after neuron count" << endl;
      return fail;
    }

    vector<Neuron> neuron_vector = readNeurons(s, neuron_count);
    if (neuron_vector.size() == 0) {
      cerr << "Error reading neurons" << endl;
      return fail;
    }
    delete fail;
    return new Layer(neuron_vector, input_size);
  }

  void Layer::setNextLayer(shared_ptr<Layer> n) {
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

  bool Layer::write(ostream &s) const {
    if (!s.good()) return false;
    s << "LAYER\n"
      << "inputs " << input_count << "\n"
      << "neurons "     << size() << "\n";
    bool success = true;
    for (vector<Neuron>::const_iterator it = neurons.begin(); it != neurons.end(); it++) {
      success &= it->write(s);
    }
    return success;
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

  vector<Neuron> Layer::readNeurons(istream &s, int count) {
    vector<Neuron> neuron_vector;
    Neuron current(0);
    for(int i = 0; i < count; i++) {
      current = Neuron::read(s);
      // Empty Neuron means something went wrong while reading Neuron data
      if (current.InputSize() == 0 || !s.good()) {
	return vector<Neuron>();
      }
      neuron_vector.push_back(current);
    }
    return neuron_vector;
  }
}
