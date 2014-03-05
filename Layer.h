#ifndef NEURAL_LAYER_H
#define NEURAL_LAYER_H

#include <memory>
#include <vector>
#include "Neuron.h"

using namespace std;

namespace neural {
  class Layer {
  public:
    Layer(int neuron_count, shared_ptr<Layer> p);
    Layer(vector<vector<double> > neuron_data, shared_ptr<Layer> p);
    Layer(int neuron_count, int i);
    Layer(vector<vector<double> > neuron_data, int i);
    void addNextLayer(shared_ptr<Layer> n);
    void updateOutputs(vector<double> inputs);
    /**
     * Recursively calculate the deltas, moving from this layer to the input layer
     * @param summedWeighedDeltas i-th element contains the summed deltas from the following layer, multiplied with the input weight corresponding to the i-th Neuron in this layer - for the output layer, just use (expected_output - actual_output)
     *                            deltas[1] = neurons[0].delta * neurons[0].weights[1]
     *                                      + neurons[1].delta * neurons[1].weights[1]
     */
    void updateDeltas(vector<double> summedWeighedDeltas);

    void updateWeights(vector<double> inputs, double learningRate);
    //! Get the number of Neurons in this layer
    inline int size() { return neurons.size(); };
    vector<double> output;
  private:
    void init_neurons(int neuron_count, int inputs);
    /**
     * init Neurons with pre-learned data
     * @param each element contains the weights for one Neuron
     */
    void init_neurons(vector<vector<double> > neuron_data);
    int input_count;
    shared_ptr<Layer> prev;
    shared_ptr<Layer> next;
    vector<Neuron> neurons;
  };
}
#endif
