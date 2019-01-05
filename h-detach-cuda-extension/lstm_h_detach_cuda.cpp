#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> h_detach_cuda_forward(
    at::Tensor input,
    at::Tensor weights,
    at::Tensor bias,
    at::Tensor old_h,
    at::Tensor old_cell);

std::vector<at::Tensor> h_detach_cuda_backward(
    at::Tensor grad_loss,
    at::Tensor grad_h,
    at::Tensor grad_cell,
    at::Tensor new_cell,
    at::Tensor old_cell,
    at::Tensor forget_gate,
    at::Tensor input_gate,
    at::Tensor output_gate,
    at::Tensor candidate_cell,
    at::Tensor X,
    at::Tensor gate_weights,
    at::Tensor weights,
    at::Tensor p_detach);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> lstm_h_detach_forward(
    at::Tensor input,
    at::Tensor weights,
    at::Tensor bias,
    at::Tensor old_h,
    at::Tensor old_cell) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);
  CHECK_INPUT(old_h);
  CHECK_INPUT(old_cell);

  return h_detach_cuda_forward(input, weights, bias, old_h, old_cell);
}

std::vector<at::Tensor> lstm_h_detach_backward(
    at::Tensor grad_loss, 
    at::Tensor grad_h,
    at::Tensor grad_cell,
    at::Tensor new_cell,
    at::Tensor old_cell,
    at::Tensor forget_gate,
    at::Tensor input_gate,
    at::Tensor output_gate,
    at::Tensor candidate_cell,
    at::Tensor X,
    at::Tensor gate_weights,
    at::Tensor weights,
    at::Tensor p_detach) {
  CHECK_INPUT(grad_loss);
  CHECK_INPUT(grad_h);
  CHECK_INPUT(grad_cell);
  CHECK_INPUT(forget_gate);
  CHECK_INPUT(input_gate);
  CHECK_INPUT(output_gate);
  CHECK_INPUT(candidate_cell);
  CHECK_INPUT(X);
  CHECK_INPUT(gate_weights);
  CHECK_INPUT(weights);
  CHECK_INPUT(p_detach);

  return h_detach_cuda_backward(
      grad_loss,
      grad_h,
      grad_cell,
      new_cell,
      old_cell,
      forget_gate,
      input_gate,
      output_gate,
      candidate_cell,
      X,
      gate_weights,
      weights,
      p_detach);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lstm_h_detach_forward, "h_detach forward (CUDA)");
  m.def("backward", &lstm_h_detach_backward, "h_detach backward (CUDA)");
}
