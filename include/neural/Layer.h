#ifndef NEURAL_LAYER_H
#define NEURAL_LAYER_H

#include <memory>
#include <vector>
#include <fstream>
#include <iostream>
#include "Neuron.h"

using namespace std;

namespace neural {

  /**
   * A single layer in a neural network. Has pointers to the previous and following 
   * layers, thus implementing a doubly-linked list. However, since there's little reason
   * to modify neural networks after creation, the list is limited to insertion at the end.
   */
  class Layer {
  public:
    Layer(int neuron_count, shared_ptr<Layer> previous);
    Layer(vector<vector<double> > neuron_data, shared_ptr<Layer> previous);
    Layer(int neuron_count, int inputs);
    Layer(vector<vector<double> > neuron_data, int inputs);
    Layer(vector<Neuron> neuron_vector, shared_ptr<Layer> previous);
    Layer(vector<Neuron> neuron_vector, int inputs);
    static Layer* read(istream &s, shared_ptr<Layer> previous);
    static Layer* read(istream &s, int input_size);
    void setNextLayer(shared_ptr<Layer> n);

    /**
     * Recursively update the outputs, moving from this layer to the output layer
     * @param inputs a vector of inputs. The first layer gets the actual input, subsequent layers get their predecessor's output vector
     */
    void updateOutputs(vector<double> inputs);

    /**
     * [Training] Recursively calculate the deltas (i.e. weighted error values), moving 
     * from this layer to the input layer.
     * @param summedWeighedDeltas i-th element contains the summed deltas from the following layer, multiplied with the input weight corresponding to the i-th Neuron in this layer -- for the output layer, just use (expected_output - actual_output). For all others:
     * <tt>deltas[1] = neurons[0].delta * neurons[0].weights[1] + neurons[1].delta * neurons[1].weights[1]</tt>
     */
    void updateDeltas(vector<double> summedWeighedDeltas);

    /**
     * [Training] Update the weights stored in each of this Layer's neurons, according to their
     * current delta values. Do this *after* Layer::updateDeltas()
     * @param inputs a vector of input values
     * @param learningRate decides how quickly the neurons' weights change during training
     */
    void updateWeights(vector<double> inputs, double learningRate);
    //! Get the number of Neurons in this layer
    inline int size() const { return neurons.size(); };

    //! Serialize this layer into \p s
    bool write(ostream &s) const;

    /**
     * Get the current output vector. Note that output is *only* updated
     *  by Layer::updateOutput, **not** implicitly by using this function!
     */
    inline vector<double> Output() { return output; };

    /**
     * Get a shared_ptr to the following Layer
     */
    inline shared_ptr<Layer> nextLayer() const { return next; };
  private:
    vector<double> output;
    void init_neurons(int neuron_count, int inputs);

    /**
     * init Neurons with pre-learned data
     * @param each element contains the weights for one Neuron
     */
    void init_neurons(vector<vector<double> > neuron_data);
    static vector<Neuron> readNeurons(istream &s, int count);
    shared_ptr<Layer> prev;
    shared_ptr<Layer> next;
    int input_count;
    vector<Neuron> neurons;
  };
}
#endif
