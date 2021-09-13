#include "ctranslate2/layers/transformer.h"
#include <spdlog/spdlog.h>
#include "type_dispatch.h"
#include "device_dispatch.h"

namespace ctranslate2
{
  namespace layers
  {

    FeedForwardNetwork::FeedForwardNetwork(const models::Model &model,
                                           const std::string &scope,
                                           const bool pre_norm,
                                           const ops::ActivationType activation_type)
        : _layer_norm(model, scope + "/layer_norm"), _pre_norm(pre_norm), _activation_type(activation_type), _ff1(model, scope + "/linear_0", &_activation_type), _ff2(model, scope + "/linear_1")
    {
    }

    void FeedForwardNetwork::operator()(const StorageView &input, StorageView &output) const
    {
      const StorageView *x = &input;
      if (_pre_norm)
      {
        _layer_norm(input, output);
        x = &output;
      }

      StorageView inner(input.dtype(), input.device());
      _ff1(*x, inner);
      _ff2(inner, output);
      ops::Add()(input, output, output);
      if (!_pre_norm)
        _layer_norm(output, output);
    }

    TransformerEncoderLayer::TransformerEncoderLayer(const models::Model &model,
                                                     const std::string &scope,
                                                     const size_t num_heads,
                                                     const bool pre_norm,
                                                     const ops::ActivationType activation_type)
        : _self_attention(model,
                          scope + "/self_attention",
                          num_heads,
                          /*self_attention=*/true,
                          pre_norm),
          _ff(model, scope + "/ffn", pre_norm, activation_type)
    {
    }

    void TransformerEncoderLayer::operator()(const StorageView &input,
                                             const StorageView &lengths,
                                             StorageView &output,
                                             const Padder *padder) const
    {
      PROFILE("TransformerEncoderLayer");
      StorageView context(input.dtype(), input.device());
      _self_attention(input, input, &lengths, context, nullptr, nullptr, nullptr, padder, padder);
      _ff(context, output);
    }

    TransformerDecoderLayer::TransformerDecoderLayer(const models::Model &model,
                                                     const std::string &scope,
                                                     const size_t num_heads,
                                                     const bool with_encoder_attention,
                                                     const bool pre_norm,
                                                     const ops::ActivationType activation_type)
        : _self_attention(model,
                          scope + "/self_attention",
                          num_heads,
                          /*self_attention=*/true,
                          pre_norm),
          _encoder_attention(with_encoder_attention
                                 ? std::make_unique<MultiHeadAttention>(model,
                                                                        scope + "/attention",
                                                                        num_heads,
                                                                        /*self_attention=*/false,
                                                                        pre_norm)
                                 : nullptr),
          _ff(model, scope + "/ffn", pre_norm, activation_type)
    {
    }

    void TransformerDecoderLayer::operator()(const StorageView &input,
                                             const StorageView *input_length,
                                             const StorageView *memory,
                                             const StorageView *memory_lengths,
                                             StorageView *cached_self_attn_keys,
                                             StorageView *cached_self_attn_values,
                                             StorageView *cached_attn_keys,
                                             StorageView *cached_attn_values,
                                             StorageView &output,
                                             StorageView *attention,
                                             const Padder *input_padder,
                                             const Padder *memory_padder) const
    {
      PROFILE("TransformerDecoderLayer");
      // spdlog::debug("in decoder transformer layer");
      _self_attention(input,
                      input,
                      input_length,
                      output,
                      cached_self_attn_keys,
                      cached_self_attn_values,
                      nullptr,
                      input_padder,
                      input_padder);
      // spdlog::debug("in decoder transformer layer 1");
      StorageView context(input.dtype(), input.device());
      if (_encoder_attention)
      {
        (*_encoder_attention)(output,
                              *memory,
                              memory_lengths,
                              context,
                              cached_attn_keys,
                              cached_attn_values,
                              attention,
                              input_padder,
                              memory_padder);
      }
      else
      {
        context = std::move(output);
      }
      // spdlog::debug("in decoder transformer layer 2");
      _ff(context, output);
    }

    static std::unique_ptr<PositionEncoder>
    build_position_encoder(const models::Model &model,
                           const std::string &scope,
                           const Embeddings &embeddings)
    {
      if (model.get_variable_if_exists(scope + "/encodings"))
        return std::make_unique<PositionEmbedding>(model, scope);
      else
        return std::make_unique<SinusoidalPositionEncoder>(embeddings.output_size(),
                                                           embeddings.output_type(),
                                                           model.device());
    }

    TransformerEncoder::TransformerEncoder(const models::Model &model,
                                           const std::string &scope,
                                           const size_t num_heads,
                                           const bool with_position_encoding,
                                           const bool pre_norm,
                                           const ops::ActivationType activation_type)
        : _embeddings(model, scope + "/embeddings"), _num_heads(num_heads), _compute_type(model.effective_compute_type()), _position_encoder(with_position_encoding
                                                                                                                                                 ? build_position_encoder(model, scope + "/position_encodings", _embeddings)
                                                                                                                                                 : nullptr),
          _output_norm(pre_norm
                           ? std::make_unique<LayerNorm>(model, scope + "/layer_norm")
                           : nullptr)
    {
      for (size_t l = 0;; ++l)
      {
        const std::string layer_scope = scope + "/layer_" + std::to_string(l);
        try
        {
          auto layer = std::make_unique<TransformerEncoderLayer>(model,
                                                                 layer_scope,
                                                                 num_heads,
                                                                 pre_norm,
                                                                 activation_type);
          _layers.emplace_back(std::move(layer));
        }
        catch (std::exception &)
        {
          if (l == 0)
            throw;
          else
            break;
        }
      }
    }

    void TransformerEncoder::operator()(const StorageView &ids,
                                        const StorageView &lengths,
                                        StorageView &output)
    {
      PROFILE("TransformerEncoder");
      StorageView input(output.dtype(), output.device());
      _embeddings(ids, input);
      if (_position_encoder)
        (*_position_encoder)(input);

      const dim_t max_time = input.dim(1);

      // Remove padding to reduce the amount of computation.
      std::unique_ptr<Padder> padder;
      if (Padder::allow_padding_removal(output.device(), _compute_type))
      {
        padder = std::make_unique<Padder>(lengths, max_time);
        padder->remove_padding(input);
      }

      const StorageView lengths_mask = layers::MultiHeadAttention::prepare_length_mask(lengths,
                                                                                       _num_heads,
                                                                                       max_time);

      for (size_t l = 0; l < _layers.size(); ++l)
      {
        (*_layers[l])(input, lengths_mask, output, padder.get());
        if (l + 1 < _layers.size())
          input = std::move(output);
      }
      if (_output_norm)
        (*_output_norm)(output, output);
      if (padder)
        padder->add_padding(output);
    }

    TransformerDecoder::TransformerDecoder(const models::Model &model,
                                           const std::string &scope,
                                           const size_t num_heads,
                                           const bool with_position_encoding,
                                           const bool with_encoder_attention,
                                           const bool pre_norm,
                                           const ops::ActivationType activation_type)
        : Decoder(model.device()), _with_encoder_attention(with_encoder_attention), _num_heads(num_heads), _compute_type(model.effective_compute_type()), _embeddings(model, scope + "/embeddings"), _position_encoder(with_position_encoding
                                                                                                                                                                                                                           ? build_position_encoder(model, scope + "/position_encodings", _embeddings)
                                                                                                                                                                                                                           : nullptr),
          _output_norm(pre_norm
                           ? std::make_unique<LayerNorm>(model, scope + "/layer_norm")
                           : nullptr),
          _proj(model, scope + "/projection")
    {
      for (size_t l = 0;; ++l)
      {
        const std::string layer_scope = scope + "/layer_" + std::to_string(l);
        try
        {
          auto layer = std::make_unique<TransformerDecoderLayer>(model,
                                                                 layer_scope,
                                                                 num_heads,
                                                                 with_encoder_attention,
                                                                 pre_norm,
                                                                 activation_type);
          _layers.emplace_back(std::move(layer));
        }
        catch (std::exception &)
        {
          if (l == 0)
            throw;
          else
            break;
        }
      }
    }

    void TransformerDecoder::set_vocabulary_mask(const StorageView &ids)
    {
      _proj.mask_weights(ids);
    }

    void TransformerDecoder::reset_vocabulary_mask()
    {
      _proj.reset_mask();
    }

    DecoderState TransformerDecoder::initial_state(bool iterative_decoding) const
    {
      DecoderState state;
      if (iterative_decoding)
      {
        const DataType dtype = output_type();
        for (size_t i = 0; i < _layers.size(); ++i)
        {
          const std::string i_str = std::to_string(i);
          state.emplace("self_keys_" + i_str, StorageView(dtype, _device));
          state.emplace("self_values_" + i_str, StorageView(dtype, _device));
          if (_with_encoder_attention)
          {
            state.emplace("memory_keys_" + i_str, StorageView(dtype, _device));
            state.emplace("memory_values_" + i_str, StorageView(dtype, _device));
          }
        }
      }
      return state;
    }

    bool TransformerDecoder::should_reorder_state(const std::string &name) const
    {
      // No need to reorder projected memory keys and values as they are the same for each beam.
      return !_with_encoder_attention || !starts_with(name, "memory");
    }

    void TransformerDecoder::operator()(dim_t step,
                                        const StorageView &ids,
                                        DecoderState &state,
                                        StorageView *logits,
                                        StorageView *attention)
    {
      return decode(ids, nullptr, step, state, logits, attention);
    }


    void TransformerDecoder::operator()(dim_t step,
                                        const StorageView &ids,
                                        const StorageView &lengths,
                                        DecoderState &state,
                                        StorageView *logits,
                                        StorageView *attention)
    {
      return decode(ids, &lengths, step, state, logits, attention);
    }

    void TransformerDecoder::operator()(const StorageView &ids,
                                        const StorageView &lengths,
                                        DecoderState &state,
                                        StorageView &logits)
    {
      return decode(ids, &lengths, 0, state, &logits);
    }

    void TransformerDecoder::decode(const StorageView &ids,
                                    const StorageView *lengths,
                                    dim_t step,
                                    DecoderState &state,
                                    StorageView *logits,
                                    StorageView *attention)
    {
      PROFILE("TransformerDecoder");
      // spdlog::debug("in transformer decoder");
      spdlog::debug("in transformer decoder {}", (dim_t)ids.shape().size());
      spdlog::debug("in transformer decoder {}", (dim_t)ids.shape().front());
      spdlog::debug("in transformer decoder {}", (dim_t)ids.shape().back());
      StorageView layer_in(output_type(), ids.device());
      StorageView layer_out(output_type(), ids.device());

      spdlog::debug("in transformer decoder 2 {}", ids.to(Device::CPU).at<int32_t>({0,0,0}));
      if ((dim_t)ids.shape().size() == (dim_t)1)
        spdlog::debug("in transformer decoder 2 {}", ids.to(Device::CPU).at<int32_t>({1}));
      _embeddings(ids, layer_in);
      spdlog::debug("in transformer decoder layer in {}", layer_in.to(Device::CPU).at<float>({0,0,0}));
      // spdlog::debug("in transformer decoder 3");
      if (layer_in.rank() == 2)
        layer_in.expand_dims(1);
      if (_position_encoder)
        if (step==0)
          (*_position_encoder)(layer_in);
        else
          (*_position_encoder)(layer_in, std::max(step, dim_t(0)));
      // spdlog::debug("in transformer after position encoder {}", layer_in.to(Device::CPU).at<float>({0,0,0}));
      // spdlog::debug("in transformer decoder 4 step {}", step);

      const dim_t max_time = layer_in.dim(1);
      const bool allow_padding_removal = Padder::allow_padding_removal(_device, _compute_type);
      // spdlog::debug("in transformer decoder 5");
      std::unique_ptr<const Padder> input_padder;
      std::unique_ptr<const StorageView> input_lengths_mask;
      if (lengths)
      {
        // spdlog::debug("in transformer decoder 5.1");
        if (false && allow_padding_removal)
        {
          // spdlog::debug("in transformer decoder 5.2");
          input_padder = std::make_unique<Padder>(*lengths, max_time);
          input_padder->remove_padding(layer_in);
        }
        // spdlog::debug("in transformer decoder 5.3");
        input_lengths_mask = std::make_unique<StorageView>(
            layers::MultiHeadAttention::prepare_length_mask(*lengths,
                                                            _num_heads,
                                                            max_time,
                                                            /*mask_future=*/true));
      }
      // spdlog::debug("in transformer decoder 6");
      StorageView *memory = nullptr;
      std::unique_ptr<const Padder> memory_padder;
      std::unique_ptr<const StorageView> memory_lengths_mask;
      // spdlog::debug("in transformer decoder 7");

      spdlog::debug(" layer start {}  {}" , layer_in.dim(0), layer_in.dim(1));
      if (_with_encoder_attention)
      {
        spdlog::debug("in transformer decoder 8");
        const StorageView &memory_lengths = state.at("memory_lengths");
        memory_lengths_mask = std::make_unique<StorageView>(
            layers::MultiHeadAttention::prepare_length_mask(memory_lengths,
                                                            _num_heads,
                                                            max_time));
        auto it = state.find("memory");
        if (it != state.end())
        {
          memory = &it->second;
          if (allow_padding_removal)
          {
            memory_padder = std::make_unique<Padder>(memory_lengths, memory->dim(1));
            memory_padder->remove_padding(*memory);
          }
        }
      }
      // spdlog::debug("in transformer decoder middle");
      for (size_t l = 0; l < _layers.size(); ++l)
      {
        StorageView *cached_self_attn_keys = nullptr;
        StorageView *cached_self_attn_values = nullptr;
        StorageView *cached_attn_keys = nullptr;
        StorageView *cached_attn_values = nullptr;

        if (step >= 0)
        {
          const std::string l_str = std::to_string(l);
          cached_self_attn_keys = &state.at("self_keys_" + l_str);
          cached_self_attn_values = &state.at("self_values_" + l_str);
          if (_with_encoder_attention)
          {
            cached_attn_keys = &state.at("memory_keys_" + l_str);
            cached_attn_values = &state.at("memory_values_" + l_str);
          }
        }

        (*_layers[l])(layer_in,
                      input_lengths_mask.get(),
                      memory,
                      memory_lengths_mask.get(),
                      cached_self_attn_keys,
                      cached_self_attn_values,
                      cached_attn_keys,
                      cached_attn_values,
                      layer_out,
                      l + 1 == _layers.size() ? attention : nullptr,
                      input_padder.get(),
                      memory_padder.get());
        layer_in = std::move(layer_out);
        //spdlog::debug(" layer shape {} {}  {} {}", l ,  layer_in.shape().size(), layer_in.shape().front(), layer_in.shape().back());
        //spdlog::debug(" layer shape 1 {}  {}", l , layer_in.dim(1));
        if (layer_in.dim(0) >1){
          spdlog::debug(" layer {}  {}", l , layer_in.scalar_at<float>({1, 0,0}));
        }
      }
      // spdlog::debug("in transformer decoder middle");
      if (step == 0)
      {
        // The memory is no longer needed as its projections were cached in the first step.
        //state.erase("memory");
      }
      // spdlog::debug("in transformer decoder middle 2");
      if (logits)
      {
        if (_output_norm)
          (*_output_norm)(layer_in, layer_in);
        if (layer_in.dim(0) >1){
          spdlog::debug(" layer _in after output_norm {}", layer_in.scalar_at<float>({1,0,0}));
        }
        const StorageView* weight = _proj._partial_weight.empty() ? &_proj._weight : &_proj._partial_weight;
        const StorageView* bias = _proj._partial_bias.empty() ? _proj._bias : &_proj._partial_bias;
        spdlog::debug(" _proj bias {}", bias->to(Device::CPU).at<float>({0}));
        spdlog::debug(" _proj weight {}", weight->to(Device::CPU).at<float>({0,0}));
        // spdlog::debug("in transformer decoder middle 3");
        _proj(layer_in, *logits);
        // spdlog::debug("in transformer decoder middle 4");
        if (step >= 0)
        {
          spdlog::debug("logits size {} {} {} {}", logits->shape().size(), logits->shape().front(), logits->shape().back(), logits->shape()[1]);
          spdlog::debug(" logits {}", logits->to(Device::CPU).scalar_at<float>({0,0,0}));
          // logits->view((*logits).data(), logits->shape());
          if (logits->shape()[1] != 1)
          {
            Shape shape = logits->shape();
            spdlog::debug(" logits {}", logits->to(Device::CPU).scalar_at<float>({0,shape[1] - 1,0}));
            auto logitsCPU = logits->to(Device::CPU);
            StorageView tmpLogits({shape.front(), shape.back()}, logitsCPU.dtype(), logitsCPU.device());
            // spdlog::debug("before assign -1 {} {} {}", shape.front(), shape[1], shape.back());
            // spdlog::debug("before assign -1 {}", tmpLogits.at<float>({0, 0}));
            for (dim_t b = 0; b < shape.front(); ++b)
            {
              for (dim_t h = 0; h < shape.back(); ++h)
              {
                // tmpLogits.at<float>({0, 0}) = 0.f;
                // TODO : here find a way to extract info from logits
                // spdlog::debug("before assign -1, {}, {}", b, h);
                // spdlog::debug("before assign -1, {}", logitsCPU.at<float>({0, 0, 0}));
                TYPE_DISPATCH(logitsCPU.dtype(),
                              tmpLogits.at<T>({b, h}) = logitsCPU.at<T>({b, shape[1] - 1, h}));
              }
            }
            // spdlog::debug("before assign");
            *logits = tmpLogits.to(logits->device());
            // spdlog::debug("after assign");
          }
          else
          {
            logits->squeeze(1);
          }
        }
        else if (input_padder)
          input_padder->add_padding(*logits);
        // spdlog::debug("in transformer decoder middle 5");
      }
      // spdlog::debug("in transformer decoder end");
    }

  }
}
