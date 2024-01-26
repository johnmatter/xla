/* Copyright 2023 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "xla/pjrt/he_pjrt_client.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/pjrt_client.h"

namespace xla {

HePjRtBuffer::HePjRtBuffer(HePjRtClient* client,
                           std::unique_ptr<PjRtBuffer> wrapped)
    : client_(client), wrapped_(std::move(wrapped)) {
  client_->TrackBuffer(this);
}

HePjRtBuffer::~HePjRtBuffer() { client_->UntrackBuffer(this); }

PjRtClient* HePjRtBuffer::client() const { return client_; }
PjRtClient* HePjRtExecutable::client() const { return client_; }

StatusOr<std::unique_ptr<PjRtBuffer>> HePjRtBuffer::CopyToDevice(
    PjRtDevice* dst_device) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> result,
                      wrapped_->CopyToDevice(dst_device));
  return std::unique_ptr<PjRtBuffer>(
      std::make_unique<HePjRtBuffer>(client_, std::move(result)));
}

HePjRtExecutable::HePjRtExecutable(
    HePjRtClient* client, std::unique_ptr<PjRtLoadedExecutable> wrapped)
    : client_(client), wrapped_(std::move(wrapped)) {}

StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
HePjRtExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<PjRtFuture<Status>>>& returned_futures) {
  std::vector<std::vector<PjRtBuffer*>> unwrapped_argument_handles;
  unwrapped_argument_handles.reserve(argument_handles.size());
  for (auto& handles : argument_handles) {
    unwrapped_argument_handles.emplace_back();
    auto& unwrapped_handles = unwrapped_argument_handles.back();
    unwrapped_handles.reserve(handles.size());
    for (PjRtBuffer* buffer : handles) {
      unwrapped_handles.push_back(
          tensorflow::down_cast<HePjRtBuffer*>(buffer)->wrapped());
    }
  }
  TF_ASSIGN_OR_RETURN(auto out, wrapped_->Execute(unwrapped_argument_handles,
                                                  options, returned_futures));
  for (auto& buffer_list : out) {
    for (std::unique_ptr<PjRtBuffer>& buffer : buffer_list) {
      buffer = std::make_unique<HePjRtBuffer>(client_, std::move(buffer));
    }
  }
  return out;
}

StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
HePjRtExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options,
    std::optional<PjRtFuture<Status>>& returned_future, bool fill_future) {
  std::vector<PjRtBuffer*> unwrapped_argument_handles;
  unwrapped_argument_handles.reserve(argument_handles.size());
  for (PjRtBuffer* buffer : argument_handles) {
    unwrapped_argument_handles.push_back(
        tensorflow::down_cast<HePjRtBuffer*>(buffer)->wrapped());
  }
  TF_ASSIGN_OR_RETURN(auto out, wrapped_->ExecuteSharded(
                                    unwrapped_argument_handles, device, options,
                                    returned_future, fill_future));
  for (std::unique_ptr<PjRtBuffer>& buffer : out) {
    buffer = std::make_unique<HePjRtBuffer>(client_, std::move(buffer));
  }
  return out;
}
StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
HePjRtExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options,
    std::optional<PjRtFuture<Status>>& returned_future, bool fill_future) {
  std::vector<PjRtBuffer*> unwrapped_argument_handles;
  unwrapped_argument_handles.reserve(argument_handles.size());
  for (PjRtBuffer* buffer : argument_handles) {
    unwrapped_argument_handles.push_back(
        tensorflow::down_cast<HePjRtBuffer*>(buffer)->wrapped());
  }
  TF_ASSIGN_OR_RETURN(auto out, wrapped_->ExecutePortable(
                                    unwrapped_argument_handles, device, options,
                                    returned_future, fill_future));
  for (std::unique_ptr<PjRtBuffer>& buffer : out) {
    buffer = std::make_unique<HePjRtBuffer>(client_, std::move(buffer));
  }
  return out;
}

HePjRtClient::HePjRtClient(std::unique_ptr<PjRtClient> wrapped)
    : wrapped_(std::move(wrapped)) {
  LOG(INFO) << "HePjRtClient created.";
  int num_mutexes = wrapped_->addressable_device_count();
  alive_buffers_ = std::vector<DeviceBuffers>(num_mutexes);
  for (int i = 0; i < num_mutexes; ++i) {
    mutex_id_from_device_id_.insert(
        {wrapped_->addressable_devices()[i]->id(), i});
  }
}

HePjRtClient::~HePjRtClient() { LOG(INFO) << "HePjRtClient destroyed."; }

StatusOr<std::unique_ptr<PjRtBuffer>> HePjRtClient::WrapBuffer(
    StatusOr<std::unique_ptr<PjRtBuffer>> to_wrap) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> buffer, std::move(to_wrap));
  return std::unique_ptr<PjRtBuffer>(
      std::make_unique<HePjRtBuffer>(this, std::move(buffer)));
}

StatusOr<std::unique_ptr<PjRtLoadedExecutable>> HePjRtClient::WrapExecutable(
    StatusOr<std::unique_ptr<PjRtLoadedExecutable>> to_wrap) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtLoadedExecutable> executable,
                      std::move(to_wrap));
  return std::unique_ptr<PjRtLoadedExecutable>(
      std::make_unique<HePjRtExecutable>(this, std::move(executable)));
}

static int GetMutexId(
    const HePjRtBuffer* buffer,
    const absl::flat_hash_map<int, int>& mutex_id_from_device_id) {
  auto iters = mutex_id_from_device_id.find(buffer->wrapped()->device()->id());
  CHECK(iters != mutex_id_from_device_id.end())
      << "Mutex id not found for device id: "
      << buffer->wrapped()->device()->id();
  return iters->second;
}

void HePjRtClient::TrackBuffer(HePjRtBuffer* buffer) {
  int mutex_id = GetMutexId(buffer, mutex_id_from_device_id_);
  {
    absl::MutexLock lock(&alive_buffers_[mutex_id].mu);
    alive_buffers_[mutex_id].alive_buffers.insert(buffer);
  }
}

void HePjRtClient::UntrackBuffer(const HePjRtBuffer* buffer) {
  if (buffer->wrapped() == nullptr) {
    return;
  }
  int mutex_id = GetMutexId(buffer, mutex_id_from_device_id_);
  {
    absl::MutexLock lock(&alive_buffers_[mutex_id].mu);
    alive_buffers_[mutex_id].alive_buffers.erase(buffer);
  }
}

void HePjRtClient::DestroyWrappedBuffersAndClient() {
  int num_mutexes = alive_buffers_.size();
  for (int i = 0; i < num_mutexes; ++i) {
    absl::MutexLock lock(&alive_buffers_[i].mu);
    for (auto* buffer : alive_buffers_[i].alive_buffers) {
      buffer->DestroyWrappedBuffer();
    }
  }
  wrapped_.reset(nullptr);
  LOG(INFO) << "HePjRtClient::DestroyWrappedBuffersAndClient completed.";
}

std::unique_ptr<HePjRtClient> HePjRtClient::CreateHePjRtClient(
    std::unique_ptr<PjRtClient> wrapped) {
  return std::make_unique<HePjRtClient>(std::move(wrapped));
}

}  // namespace xla

