#ifndef NEURAL_ACTIVATION_H
#define NEURAL_ACTIVATION_H

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

    enum Functions {
      SIGMOID,
      TANH
    };
  }
}
#endif
