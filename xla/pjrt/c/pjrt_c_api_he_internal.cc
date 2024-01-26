/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/pjrt/c/pjrt_c_api_he_internal.h"

#include <memory>
#include <utility>

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/he_pjrt_client.h"

namespace pjrt {
namespace he_plugin {

PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Client_Create_Args", PJRT_Client_Create_Args_STRUCT_SIZE,
      args->struct_size));

  PJRT_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtClient> client, xla::GetHeClient());
  args->client = pjrt::CreateWrapperClient(std::move(client));
  return nullptr;
}

constexpr PJRT_Api pjrt_api = pjrt::CreatePjrtApi(he_plugin::PJRT_Client_Create);

const PJRT_Api* GetHePjrtApi() { return &pjrt_api; }

}  // namespace cpu_plugin
}  // namespace pjrt
