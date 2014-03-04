#include "Neuron.h"
#include <cstdlib>
#include <assert.h>

namespace neural {
  Neuron::Neuron(int inputSize, 
		 std::function<double (double)> activationFunc,
		 std::function<double (double)> derivFunc) :
    activationFunction(activationFunc),
    derivFunction(derivFunc)
 {
    activationFunction = activationFunc;
    // NOT an off-by-one error, additional element is bias weight (bias input is always 1)
    for (int i = 0; i < inputSize + 1; i++) {
      weights.push_back((((double) rand()) / ((double) (RAND_MAX/2))) - 1); // Random value between -0.5 and 0.5
    }
  }

  void Neuron::updateOutput(std::vector<double> inputs) {
    assert(inputs.size() + 1 == weights.size());
    output = weights.back(); // bias
    for (int i = 0; i < inputs.size(); i++) {
      output += inputs[i] * weights[i];
    }
  }
  
  void Neuron::updateDelta(double deltaSum) {
    delta = derivFunction(output) * deltaSum;
  }

  void Neuron::updateWeights(std::vector<double> input, double learningRate) {
    assert(input.size() + 1 == weights.size());
    for(int i = 0; i < weights.size() - 1; i++) {
      weights[i] = input[i] * delta * learningRate;
    }
    // bias
    weights[weights.size() - 1] = delta * learningRate;
  }
}
