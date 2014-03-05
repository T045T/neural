#ifndef NEURAL_NEURON_H
#define NEURAL_NEURON_H

#include <vector>
#include <functional>
#include <cmath>

namespace neural {
  namespace activation {
    const std::function<double (double)> sigmoid_func = 
      [](double in ) -> double {
      if (in < -45.0) {
	return 0.0;
      } else if (in > 45.0) {
	return 1.0;
      } else {
	return 1.0 / (1 + exp(-in));
      }
    };

    const std::function<double (double)> sigmoid_deriv =
      [](double in) -> double {
      return in * (1 - in);
    };

    const std::function<double (double)> tanh_func =
      [](double in) -> double {
      if (in < -10.0) {
	return -1.0;
      } else if (in > 10.0) {
	return 1.0;
      } else {
	return tanh(in);
      }
    };

    const std::function<double (double)> tanh_deriv =
      [](double in) -> double {
      return (1 + in) * (1-in);
    };
  }
  class Neuron {
  public:
    /**
     * Create a new Neuron
     * @param inputSize how many inputs the Neuron has, not including its bias
     * @param activationFunc a lambda or function pointer describing the Neuron's
     *                       activation function - defaults to sigmoid_func
     * @param derivFunc a lambda or function pointer describing the derivative of
     *                 activationFunc, except input is activationFunc(x) rather 
     *                 than x - this lets us pass Output() to derivFunc directly
     *                  - defaults to sigmoid_deriv
     */
    Neuron(int inputSize, 
	   std::function<double (double)> activationFunc = activation::sigmoid_func,
	   std::function<double (double)> derivFunc = activation::sigmoid_deriv
	   );

    /**
     * Create a neuron with specific weights (including one for its bias!)
     * @param weights the weights for this neuron, in order - last one is for the bias
     * @param activationFunc a lambda or function pointer describing the Neuron's
     *                       activation function - defaults to sigmoid_func
     * @param derivFunc a lambda or function pointer describing the derivative of
     *                 activationFunc, except input is activationFunc(x) rather 
     *                 than x - this lets us pass Output() to derivFunc directly
     *                  - defaults to sigmoid_deriv
     */
    Neuron(std::vector<double> weights, 
	   std::function<double (double)> activationFunc = activation::sigmoid_func,
	   std::function<double (double)> derivFunc = activation::sigmoid_deriv
	   );

    /**
     * Update this Neuron's output value - use Output() to access it
     * @param inputs the input values for this neuron (typically outputs of all Neurons in the previous Layer)
     */
    void updateOutput(std::vector<double> inputs);
    inline double Output() const { return output; }

    /**
     * Update the delta value
     * @param deltaSum the summed deltas of the following layer, weighed by the input weight corresponding to this Neuron
     *                 f.i. (this Neuron has Index 0 in its layer, next layer has 2 Neurons):
     *                 deltaSum = next_layer.neurons[0].delta * next_layer.neurons[0].weights[0]
     *                          + next_layer.neurons[1].delta * next_layer.neurons[1].weights[0]
     */
    void updateDelta(double deltaSum);
    /**
     * Get current delta (need to call updateDelta() first!)
     * @param wrt (with-regard-to) optionally multiply delta with weight for one of this Neuron's inputs
     */
    inline double Delta(int wrt = -1) const {
      if (wrt < 0 || wrt >= weights.size() - 1) {
	return delta;
      } else {
	return delta * weights[wrt];
      }
    }

    void updateWeights(std::vector<double> input, double learningRate);
  protected:
    std::function<double (double)> activationFunction;
    std::function<double (double)> derivFunction;
    //! Size will be inputSize + 1 - the additional item is the bias
    std::vector<double> weights;
    double output;
    double delta;
  };
}
#endif
