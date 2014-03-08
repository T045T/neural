#ifndef NEURAL_LAYER_H
#define NEURAL_LAYER_H

#include <memory>
#include <vector>
#include <fstream>
#include <iostream>
#include "Neuron.h"

using namespace std;

namespace neural {
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
    inline int size() const { return neurons.size(); };
    bool write(ostream &s) const;
    inline vector<double> Output() { return output; };
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
