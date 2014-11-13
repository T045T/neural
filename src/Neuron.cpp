#include "neural/Neuron.h"
#include <cstdlib>
#include <cassert>

namespace neural {
  Neuron::Neuron(int inputSize, 
		 std::function<double (double)> activation_func,
		 std::function<double (double)> deriv_func) :
    activationFunction(activation_func),
    derivFunction(deriv_func)
 {
   initWeightsRandom(inputSize);
 }

  Neuron::Neuron(std::vector<double> w, 
		 std::function<double (double)> activation_func,
		 std::function<double (double)> deriv_func) :
    activationFunction(activation_func),
    derivFunction(deriv_func)
  {
    initWeights(w);
  }

  Neuron Neuron::read(std::istream &s, 
		      std::function<double (double)> activation_func,
		      std::function<double (double)> deriv_func)
  {
    if (!s.good()) return Neuron(0);
    std::string keyword;
    s >> keyword;
    if(keyword != "NEURON") return Neuron(0);

    s >> keyword;
    if (keyword != "size") return Neuron(0);
    int dataSize;
    s >> dataSize;

    s >> keyword;
    if (keyword != "data") return Neuron(0);
    // consume space after "data"
    char c = s.get();
    if (c != ' ') return Neuron(0);
    std::vector<double> w;
    double current = 0;
    for (int i = 0; i < dataSize; i++) {
      s.read(reinterpret_cast<char*>( &current), sizeof(current));
      w.push_back(current);
    }
    // consume ending newline
    do {
      c = s.get();
    } while (s.good() && c != '\n');

    return Neuron(w, activation_func, deriv_func);
  }

  void Neuron::updateOutput(std::vector<double> inputs) {
    assert(inputs.size() + 1 == weights.size());
    output = weights.back(); // bias
    for (int i = 0; i < inputs.size(); i++) {
      output += inputs[i] * weights[i];
    }
    output = activationFunction(output);
  }
  
  void Neuron::updateDelta(double delta_sum) {
    delta = derivFunction(output) * delta_sum;
  }

  void Neuron::updateWeights(std::vector<double> input, double learning_rate) {
    assert(input.size() + 1 == weights.size());
    for(int i = 0; i < weights.size() - 1; i++) {
      weights[i] += input[i] * delta * learning_rate;
    }
    // bias
    weights[weights.size() - 1] = delta * learning_rate;
  }

  bool Neuron::write(std::ostream &s) const {
    assert(sizeof(double) == sizeof(char*));
    if ( !s.good() ) return false;
    s << "NEURON" << "\n"
      << "size " << weights.size() << "\n"
      << "data ";
    double current = 0.0;
    for ( int i = 0; i < weights.size(); i++) {
      current = weights[i];
      s.write(reinterpret_cast<char*>(&current), sizeof(current));
      if (!s.good()) {
	return false;
      }
    }
    s << std::endl;
    return true;
  }

  void Neuron::initWeightsRandom(int inputSize) {
    for (int i = 0; i < inputSize + 1; i++) {
      weights.push_back((((double) rand()) / ((double) (RAND_MAX/2))) - 1); // Random value between -0.5 and 0.5
    }
  }

  void Neuron::initWeights(std::vector<double> w) {
    for (std::vector<double>::iterator it = w.begin(); it != w.end(); it++) {
      weights.push_back(*it);
    }
  }
}
