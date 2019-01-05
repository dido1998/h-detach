#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

#include <curand_kernel.h>
#include "device_launch_parameters.h"
#include <ctime>

#include <stdio.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (1.0 - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
  const auto t = tanh(z);
  return 1 - (t * t);
}

template <typename scalar_t>
__global__ void h_detach_cuda_forward_kernel(
    const scalar_t* __restrict__ gates,
    const scalar_t* __restrict__ old_cell,
    scalar_t* __restrict__ new_h,
    scalar_t* __restrict__ new_cell,
    scalar_t* __restrict__ forget_gate,
    scalar_t* __restrict__ input_gate,
    scalar_t* __restrict__ output_gate,
    scalar_t* __restrict__ candidate_cell,
    size_t state_size) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int index = blockIdx.y * state_size + column;
  const int gates_row = blockIdx.y * (state_size * 4);
  if (column < state_size) {
    forget_gate[index]=sigmoid(gates[gates_row+column]);
    input_gate[index] = sigmoid(gates[gates_row + state_size+column]);
    output_gate[index] = sigmoid(gates[gates_row + 2*state_size + column]);
    candidate_cell[index] = tanh(gates[gates_row + 3 * state_size + column]);
    new_cell[index] =old_cell[index]*forget_gate[index] + candidate_cell[index] * input_gate[index];
    new_h[index] = tanh(new_cell[index]) * output_gate[index];
  }
}
std::vector<at::Tensor> h_detach_cuda_forward(
    at::Tensor input,
    at::Tensor weights,
    at::Tensor bias,
    at::Tensor old_h,
    at::Tensor old_cell) {
  auto X = at::cat({old_h, input}, /*dim=*/1);
  auto gates = at::addmm(bias, X, weights.transpose(0, 1));
  const auto batch_size=old_cell.size(0);
  const auto state_size = old_cell.size(1);

  auto new_h = at::zeros_like(old_cell);
  auto new_cell = at::zeros_like(old_cell);
  auto forget_gate = at::zeros_like(old_cell);
  auto input_gate = at::zeros_like(old_cell);
  auto output_gate = at::zeros_like(old_cell);
  auto candidate_cell = at::zeros_like(old_cell);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(gates.type(), "h_detach_forward_cuda", ([&] {
    h_detach_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        gates.data<scalar_t>(),
        old_cell.data<scalar_t>(),
        new_h.data<scalar_t>(),
        new_cell.data<scalar_t>(),
        forget_gate.data<scalar_t>(),
        input_gate.data<scalar_t>(),
        output_gate.data<scalar_t>(),
        candidate_cell.data<scalar_t>(),
        state_size);
  }));

  return {new_h, new_cell, old_cell,forget_gate,input_gate, output_gate, candidate_cell, X, gates};
}





template <typename scalar_t>
__global__ void h_detach_cuda_backward_kernel(
    scalar_t* __restrict__ d_old_cell,
    scalar_t* __restrict__ d_gates,
    const scalar_t* __restrict__ grad_loss,
    scalar_t* __restrict__ grad_h,
    const scalar_t* __restrict__ grad_cell,
    const scalar_t* __restrict__ new_cell,
    const scalar_t* __restrict__ old_cell,
    const scalar_t* __restrict__ forget_gate,
    const scalar_t* __restrict__ input_gate,
    const scalar_t* __restrict__ output_gate,
    const scalar_t* __restrict__ candidate_cell,
    const scalar_t* __restrict__ gate_weights,
    const double* __restrict__ p_detach,
    size_t state_size,
    double p) {


    


  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int index = blockIdx.y * state_size + column;
  const int gates_row = blockIdx.y * (state_size * 4);
  
  //curandState state;
  //curand_init((unsigned long long)clock() + column, 0, 0, &state);

   //double p = curand_uniform_double(&state);


  if (column < state_size) {
    //if (p<*p_detach)
    //{
    //grad_h[index]=0;
    //}
    const auto d_output_gate = tanh(new_cell[index]) * (grad_h[index]+grad_loss[index]);
    const auto d_tanh_new_cell = output_gate[index] * (grad_h[index]+grad_loss[index]);
    
    const auto d_new_cell =d_tanh(new_cell[index]) * d_tanh_new_cell + grad_cell[index];
    d_old_cell[index]=forget_gate[index]*d_new_cell;
    const auto d_forget_gate=old_cell[index]*d_new_cell;
    const auto d_candidate_cell = input_gate[index] * d_new_cell;
    const auto d_input_gate = candidate_cell[index] * d_new_cell;

    const auto forget_gate_index = gates_row + column;
    const auto input_gate_index = gates_row +state_size+ column;
    const auto output_gate_index = gates_row + 2*state_size + column;
    const auto candidate_cell_index = gates_row + 3 * state_size + column;

    d_gates[forget_gate_index]=
      d_forget_gate*d_sigmoid(gate_weights[forget_gate_index]);
   d_gates[input_gate_index] =
        d_input_gate * d_sigmoid(gate_weights[input_gate_index]);
    d_gates[output_gate_index] =
        d_output_gate * d_sigmoid(gate_weights[output_gate_index]);
    d_gates[candidate_cell_index] =
        d_candidate_cell * d_tanh(gate_weights[candidate_cell_index]);
  }
}

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
    at::Tensor p_detach) {
  auto d_old_cell = at::zeros_like(new_cell);
  auto d_gates = at::zeros_like(gate_weights);

  const auto batch_size = new_cell.size(0);
  const auto state_size = new_cell.size(1);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);
  srand(static_cast<unsigned>(time(0)));
  double p=rand()%100/99.0;
  AT_DISPATCH_FLOATING_TYPES(X.type(), "h_detach_backward_cuda", ([&] {
    h_detach_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        d_old_cell.data<scalar_t>(),
        d_gates.data<scalar_t>(),
        grad_loss.contiguous().data<scalar_t>(),
        grad_h.contiguous().data<scalar_t>(),
        grad_cell.contiguous().data<scalar_t>(),
        new_cell.contiguous().data<scalar_t>(),
        old_cell.contiguous().data<scalar_t>(),
        forget_gate.contiguous().data<scalar_t>(),
        input_gate.contiguous().data<scalar_t>(),
        output_gate.contiguous().data<scalar_t>(),
        candidate_cell.contiguous().data<scalar_t>(),
        gate_weights.contiguous().data<scalar_t>(),
        p_detach.data<double>(),
        state_size,
        p);
  }));

  auto d_weights=d_gates.t().mm(X);
  auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);


  auto d_X = d_gates.mm(weights);
  auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
}
