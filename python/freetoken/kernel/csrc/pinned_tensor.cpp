#include <cstdint>
#include <hip/hip_runtime_api.h>
#include <torch/extension.h>

namespace {

void free_pinned(void *ptr) {
  if (ptr != nullptr) {
    hipHostFree(ptr);
  }
}

torch::Tensor create_pinned_tensor_like(torch::Tensor input) {
  TORCH_CHECK(input.device().is_cpu(), "Input tensor must be on CPU");
  TORCH_CHECK(input.layout() == torch::kStrided,
              "Input tensor must have strided layout");

  const auto sizes = input.sizes().vec();
  const auto strides = input.strides().vec();
  const int64_t itemsize = input.element_size();
  TORCH_CHECK(itemsize > 0, "Input tensor element size must be positive");

  const bool is_empty = input.numel() == 0;
  uint64_t storage_elements = is_empty ? 0 : 1;
  for (int64_t i = 0; i < static_cast<int64_t>(sizes.size()); ++i) {
    TORCH_CHECK(strides[i] >= 0, "Negative strides are not supported");
    if (!is_empty) {
      storage_elements += static_cast<uint64_t>(sizes[i] - 1) *
                          static_cast<uint64_t>(strides[i]);
    }
  }

  const uint64_t nbytes = storage_elements * static_cast<uint64_t>(itemsize);
  const size_t alloc_nbytes = static_cast<size_t>(nbytes == 0 ? 1 : nbytes);

  void *data_ptr = nullptr;
  const hipError_t alloc_err = hipHostMalloc(&data_ptr, alloc_nbytes);
  TORCH_CHECK(alloc_err == hipSuccess,
              "hipHostMalloc failed: ", hipGetErrorString(alloc_err));

  auto options = input.options().device(torch::kCPU).pinned_memory(true);

  return torch::from_blob(data_ptr, sizes, strides, free_pinned, options);
}

torch::Tensor alloc_pinned_tensor(std::vector<int64_t> sizes,
                                  at::ScalarType dtype) {
  int64_t numel = 1;
  for (const int64_t s : sizes) {
    TORCH_CHECK(s >= 0, "Sizes must be non-negative");
    numel *= s;
  }

  const uint64_t nbytes =
      static_cast<uint64_t>(numel) * c10::elementSize(dtype);
  const size_t alloc_nbytes = static_cast<size_t>(nbytes == 0 ? 1 : nbytes);

  // Portable + mapped: the offload gather kernel reads these banks straight
  // from host memory (zero-copy), which requires device-mapped pinned pages.
  void *data_ptr = nullptr;
  const hipError_t alloc_err = hipHostAlloc(
      &data_ptr, alloc_nbytes, hipHostMallocPortable | hipHostMallocMapped);
  TORCH_CHECK(alloc_err == hipSuccess,
              "hipHostAlloc failed: ", hipGetErrorString(alloc_err));

  auto options = torch::TensorOptions()
                     .dtype(dtype)
                     .device(torch::kCPU)
                     .pinned_memory(true);

  return torch::from_blob(data_ptr, sizes, free_pinned, options);
}

// Pinned host memory is GPU-dereferenceable at its host VA only where UVA identity
// holds (Linux; not Windows/WDDM, where hipHostRegister'd memory maps to a different
// device address). Zero-copy consumers resolve bank base addresses through these.
bool host_ptr_identity() {
  int device = 0;
  const hipError_t err = hipGetDevice(&device);
  TORCH_CHECK(err == hipSuccess, "hipGetDevice failed: ", hipGetErrorString(err));
  int uva = 0, reg = 0;
  hipDeviceGetAttribute(&uva, hipDeviceAttributeUnifiedAddressing, device);
  hipDeviceGetAttribute(&reg, hipDeviceAttributeCanUseHostPointerForRegisteredMem, device);
  return uva == 1 && reg == 1;
}

int64_t host_device_ptr(int64_t host_ptr) {
  void *dev_ptr = nullptr;
  const hipError_t err =
      hipHostGetDevicePointer(&dev_ptr, reinterpret_cast<void *>(host_ptr), 0);
  TORCH_CHECK(err == hipSuccess,
              "hipHostGetDevicePointer failed (host memory must be pinned+mapped): ",
              hipGetErrorString(err));
  return reinterpret_cast<int64_t>(dev_ptr);
}

void host_register(int64_t addr, int64_t nbytes) {
  const hipError_t err =
      hipHostRegister(reinterpret_cast<void *>(addr), static_cast<size_t>(nbytes),
                       hipHostRegisterPortable | hipHostRegisterMapped);
  TORCH_CHECK(err == hipSuccess,
              "hipHostRegister failed: ", hipGetErrorString(err));
}

// Give a registration back. The pages stay where they are and stay readable by everyone;
// what is released is this process's device mapping of them and its claim on the
// machine-wide registration budget -- which is the whole point, because that budget is
// what stops a second process from mapping the same pages for its own device.
void host_unregister(int64_t addr) {
  hipError_t err = hipHostUnregister(reinterpret_cast<void *>(addr));
  TORCH_CHECK(err == hipSuccess, "hipHostUnregister failed: ", hipGetErrorString(err));
}

int64_t driver_cuda_version() {
  int version = 0;  // stays 0 when no driver is installed
  const hipError_t err = hipDriverGetVersion(&version);
  TORCH_CHECK(err == hipSuccess,
              "hipDriverGetVersion failed: ", hipGetErrorString(err));
  return version;
}

} // namespace

// A device tensor naming memory the allocator never handed out.
//
// Registered host memory has a device address, and on this platform it is the same address
// the CPU uses -- so an accelerator can read those pages in place, with no copy and no
// second residency. What it does not have is a tensor: every kernel here takes one, and
// there is no way to build one over a bare address from Python. This is that way.
//
// The storage is not owned. Nothing is freed when the tensor dies, because the memory
// belongs to whoever registered it -- the caller keeps it alive for as long as the tensor
// is used, exactly as it already does for the mapping itself.
static torch::Tensor tensor_from_device_ptr(uintptr_t addr, std::vector<int64_t> sizes,
                                            py::object dtype, int64_t device_index) {
  auto scalar_type = torch::python::detail::py_object_to_dtype(std::move(dtype));
  auto device = c10::Device(torch::kCUDA, (c10::DeviceIndex)device_index);
  auto options = torch::TensorOptions().dtype(scalar_type).device(device);
  // target_device, not from_blob's inference. Asked to infer, ATen reads the pointer's
  // attributes, and for registered host pages the runtime answers with the device the
  // registration was made for -- so a second device of this process, reading the very same
  // portable mapping, is refused as "does not match device of data". The caller knows which
  // device is about to dereference this; say so instead of asking.
  return at::for_blob(reinterpret_cast<void*>(addr), sizes)
      .deleter([](void*) {})
      .options(options)
      .target_device(device)
      .make_tensor();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("create_pinned_tensor_like", &create_pinned_tensor_like,
        "Create an exact-size CPU pinned tensor with input's size/stride/dtype");
  m.def("alloc_pinned_tensor", &alloc_pinned_tensor,
        "Allocate an exact-size, uninitialized CPU pinned tensor");
  m.def("host_ptr_identity", &host_ptr_identity,
        "True if the GPU dereferences pinned host memory at its host VA (UVA identity)");
  m.def("host_device_ptr", &host_device_ptr,
        "Device-visible alias of a pinned+mapped host address");
  m.def("tensor_from_device_ptr", &tensor_from_device_ptr,
        "A device tensor over memory this extension did not allocate",
        py::arg("addr"), py::arg("sizes"), py::arg("dtype"), py::arg("device_index"));
  m.def("host_unregister", &host_unregister,
        "hipHostUnregister a range registered by host_register");
  m.def("host_register", &host_register,
        "hipHostRegister an existing host range as portable+mapped");
  m.def("driver_cuda_version", &driver_cuda_version,
        "Max CUDA version the installed NVIDIA driver supports (0 if none)");
}
