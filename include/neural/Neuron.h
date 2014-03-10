#ifndef NEURAL_NEURON_H
#define NEURAL_NEURON_H

#include <vector>
#include <functional>
#include <cmath>
#include <fstream>
#include <iostream>

#include "Activation.h"

namespace neural {
  class Neuron {
  public:

    /**
     * Create a new Neuron
     * @param inputSize how many inputs the Neuron has, not including its bias
     * @param activationFunc a lambda or function pointer describing the Neuron's
     *                       activation function - defaults to tanh_func
     * @param derivFunc a lambda or function pointer describing the derivative of
     *                 activationFunc, except input is activationFunc(x) rather 
     *                 than x - this lets us pass Output() to derivFunc directly
     *                  - defaults to tanh_deriv
     */
    Neuron(int inputSize, 
	   std::function<double (double)> activationFunc = activation::tanh_func,
	   std::function<double (double)> derivFunc = activation::tanh_deriv
	   );

    /**
     * Create a neuron with specific weights (including one for its bias!)
     * @param weights the weights for this neuron, in order - last one is for the bias
     * @param activationFunc a lambda or function pointer describing the Neuron's
     *                       activation function - defaults to tanh_func
     * @param derivFunc a lambda or function pointer describing the derivative of
     *                 activationFunc, except input is activationFunc(x) rather 
     *                 than x - this lets us pass Output() to derivFunc directly
     *                  - defaults to tanh_deriv
     */
    Neuron(std::vector<double> w, 
	   std::function<double (double)> activationFunc = activation::tanh_func,
	   std::function<double (double)> derivFunc = activation::tanh_deriv
	   );

    /**
     * Construct Neuron from input stream - defaults to tanh activation function
     */
    static Neuron read(std::istream &s, 
	   std::function<double (double)> activationFunc = activation::tanh_func,
	   std::function<double (double)> derivFunc = activation::tanh_deriv
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

    inline int InputSize() { return weights.size() - 1; };
    /**
     * Update the weights using the delta calculated with @see updateDelta()
     * @param input the input to this layer
     * @param learningRate the change in weight will be multiplied with this
     *                     higher values mean faster convergence, but are more likely
     *                     to overshoot the actual minimum
     */
    void updateWeights(std::vector<double> input, double learningRate);

    /**
     * Write this Neuron's data to an output stream. Size is written in ASCII, weights 
     * are stored as a binary blob - the activation function isn't saved at all!
     *
     * @return true if writing to the stream was successfull, false if it was not
     */
    bool write(std::ostream &s) const;

  protected:
     //! Initialize this Neuron's weights to random values between -0.5 and 0.5 - there will be one more weight than @param inputSize because of the bias
    void initWeightsRandom(int inputSize);

    /**
     * Initialize this Neuron's weights (including the bias weight) to the given values
     * @param w a vector containing all weights for this neuron, including the bias weight
     */
    void initWeights(std::vector<double> w);
    std::function<double (double)> activationFunction;
    std::function<double (double)> derivFunction;
    //! Size will be inputSize + 1 - the additional item is the bias
    std::vector<double> weights;
    double output;
    double delta;
  };
}
#endif
