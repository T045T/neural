#ifndef NEURAL_LAYER_H
#define NEURAL_LAYER_H

#include <memory>
#include <vector>
#include "Neuron.h"

namespace neural {
  class Layer {
  public:
    Layer(int neurons, shared_ptr<Layer> prev = shared_ptr<Layer>());
    void addNextLayer(shared_ptr<Layer> next);
    bool updateOutputs(vector<double> inputs);
    /**
     * Calculate the deltas in this layer
     * @param summedWeighedDeltas i-th element contains the summed deltas from the following layer, multiplied with the input weight corresponding to the i-th Neuron in this layer - for the output layer, just use (expected_output - actual_output)
     *                            deltas[1] = neurons[0].delta * neurons[0].weights[1]
     *                                      + neurons[1].delta * neurons[1].weights[1]
     */
    bool updateDeltas(vector<double> summedWeighedDeltas);

    void updateWeights(vector<double> inputs, double learningRate);
    vector<double> Output();
  private:
    shared_ptr<Layer> prev;
    shared_ptr<Layer> next;
    vector<Neuron> neurons;
  };
}
#endif
