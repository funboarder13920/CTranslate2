#include "ctranslate2/layers/decoder.h"

#include "ctranslate2/ops/ops.h"
#include <spdlog/spdlog.h>

namespace ctranslate2 {
  namespace layers {

    Decoder::Decoder(Device device)
      : _device(device) {
    }

    void Decoder::gather_state(DecoderState& state, const StorageView& indices) const {
      static const ops::Gather gather_op;
      // spdlog::debug("inside gather state"); 
      // When the batch size is unchanged, assume that we are reordering beams.
      bool beam_reordering = indices.size() == batch_size(state);
// spdlog::debug("inside gather state 1"); 
      for (auto& pair : state) {
        const auto& name = pair.first;
        auto& value = pair.second;
        if (beam_reordering && !should_reorder_state(name))
          continue;
          // spdlog::debug("inside gather state 2 {} {} {}", name, value.size(), indices.size()); 
        gather_op(value, indices);
        // spdlog::debug("inside gather state 3 "); 
      }
      // spdlog::debug("inside gather state 4"); 
    }

    dim_t Decoder::batch_size(const DecoderState& state) const {
      return state.begin()->second.dim(0);
    }

    bool Decoder::should_reorder_state(const std::string&) const {
      return true;
    }

    Device Decoder::device() const {
      return _device;
    }

  }
}
