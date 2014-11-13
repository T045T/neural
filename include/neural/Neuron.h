#ifndef NEURAL_NEURON_H
#define NEURAL_NEURON_H

#include <vector>
#include <functional>
#include <cmath>
#include <fstream>
#include <iostream>

#include "Activation.h"

namespace neural {
  /**
   * This class models a single neuron in one of the @link Layer Layers @endlink that make up a Network
   *
   * Each Neuron holds a set of input weights, an additional bias weight, and an activation function
   * along with that function's derivative.
   */
  class Neuron {
  public:

    /**
     * Create a new Neuron
     * @param inputSize how many inputs the Neuron has, not including its bias
     * @param activationFunc a lambda or function pointer describing the Neuron's
     *                       activation function -- defaults to tanh_func
     * @param derivFunc a lambda or function pointer describing the derivative of
     *                 \p activationFunc, except input is \p activationFunc(x) rather 
     *                 than x -- this lets us pass Neuron::Output() to \p derivFunc directly
     *                 (defaults to tanh_deriv)
     */
    Neuron(int inputSize, 
	   std::function<double (double)> activationFunc = activation::tanh_func,
	   std::function<double (double)> derivFunc = activation::tanh_deriv
	   );

    /**
     * Create a neuron with specific weights (including one for its bias!)
     * @param weights the weights for this neuron, in order -- last one is for the bias
     * @param activationFunc a lambda or function pointer describing the Neuron's
     *                       activation function -- defaults to tanh_func
     * @param derivFunc a lambda or function pointer describing the derivative of
     *                 \p activationFunc, except input is \p activationFunc(x) rather 
     *                 than x -- this lets us pass Neural::Output() to \p derivFunc directly
     *                 (defaults to tanh_deriv)
     */
    Neuron(std::vector<double> w, 
	   std::function<double (double)> activationFunc = activation::tanh_func,
	   std::function<double (double)> derivFunc = activation::tanh_deriv
	   );

    /**
     * Construct Neuron from input stream \p s with the given activation function
     * (defaults to tanh) -- if anything goes wrong,
     * this will return a Neuron with an InputSize() of 0!
     */
    static Neuron read(std::istream &s, 
	   std::function<double (double)> activationFunc = activation::tanh_func,
	   std::function<double (double)> derivFunc = activation::tanh_deriv
	   );
    
    /**
     * Update this Neuron's output value -- use Neuron::Output() to access it
     * @param inputs the input values for this neuron (typically outputs of all neurons in the previous Layer)
     */
    void updateOutput(std::vector<double> inputs);

    /**
     * Get the current output value -- this is only updated by Neuron::updateOutput(), *not* automatically!
     */
    inline double Output() const { return output; }

    /**
     * Update the delta value
     *
     * This is the back-propagation this type of neural network gets its name from.
     * @param deltaSum the summed deltas of the following layer, weighed by the input weight corresponding to this Neuron.
     *   Example (this Neuron has Index 0 in its layer, next layer has 2 neurons):
     *   \f$\text{deltaSum} = \text{next_layer.neurons[0].delta} * \text{next_layer.neurons[0].weights[0]}
     *                    + \text{next_layer.neurons[1].delta} * \text{next_layer.neurons[1].weights[0]}\f$
     */
    void updateDelta(double deltaSum);

    /**
     * Get current delta (need to call Neuron::updateDelta() first!)
     * @param wrt (with-regard-to) optionally multiply delta with weight for one of this Neuron's inputs
     */
    inline double Delta(int wrt = -1) const {
      if (wrt < 0 || wrt >= weights.size() - 1) {
	return delta;
      } else {
	return delta * weights[wrt];
      }
    }
    
    /**
     * Get the number of input weights for this Neuron -- this is equal to the number of neurons in the previous layer
     */
    inline int InputSize() { return weights.size() - 1; };
    
    /**
     * Update the weights using the delta calculated with Neuron::updateDelta()
     * @param input the input to this layer
     * @param learningRate the change in weight will be multiplied with this --
     *                     higher values mean faster convergence, but are more likely
     *                     to overshoot the actual minimum.
     */
    void updateWeights(std::vector<double> input, double learningRate);

    /**
     * Write this Neuron's data to an output stream. Size is written in ASCII, weights 
     * are stored as a binary blob -- the activation function isn't saved at all!
     *
     * @return true if writing to the stream was successfull, false if it was not
     */
    bool write(std::ostream &s) const;

  protected:
    /**
     * Initialize this Neuron's weights to random values between -0.5 and 0.5 -- there will be one more 
     * weight than \p inputSize to account for the Neuron's bias
     */
    void initWeightsRandom(int inputSize);

    /**
     * Initialize this Neuron's weights (including the bias weight) to the given values
     * @param w a vector containing all weights for this neuron, including the bias weight
     */
    void initWeights(std::vector<double> w);
    std::function<double (double)> activationFunction;
    std::function<double (double)> derivFunction;
    //! Size will be inputSize + 1 -- the additional item is the bias
    std::vector<double> weights;
    double output;
    double delta;
  };
}
#endif
