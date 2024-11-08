"What to do when you don't know anything about the functions and classes in a module and don't have access to documentation"

from transformers import AutoTokenizer, AutoModelForCausalLM

"Running example: "
model_path = 'meta-llama/Llama-3.1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

"""
// help(<>) will give you a brief description of the class/attribute.
// Does not help with built in methods.
// Note: When in pdb mode, help() only works in interactive mode, which can be enabled by simply typing "interact" in the pdb console.
// To exit from interactive mode, type "ctrl + d".
"""
help(model)
"""
Help on LlamaForCausalLM in module transformers.models.llama.modeling_llama object:

class LlamaForCausalLM(LlamaPreTrainedModel)
 |  LlamaForCausalLM(config)
 |
 |  Method resolution order:
 |      LlamaForCausalLM
 |      LlamaPreTrainedModel
 |      transformers.modeling_utils.PreTrainedModel
 |      torch.nn.modules.module.Module
 |      transformers.modeling_utils.ModuleUtilsMixin
 |      transformers.generation.utils.GenerationMixin
 |      transformers.utils.hub.PushToHubMixin
 |      transformers.integrations.peft.PeftAdapterMixin
 |      builtins.object
 |
 |  Methods defined here:
 |
 |  __init__(self, config)
 |      Initialize internal Module state, shared by both nn.Module and ScriptModule.
 |
 |  forward(self, input_ids: torch.LongTensor = None, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, past_key_values: Union[transformers.cache_utils.Cache, List[torch.FloatTensor], NoneTyp
e] = None, inputs_embeds: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None, use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict
: Optional[bool] = None, cache_position: Optional[torch.LongTensor] = None) -> Union[Tuple, transformers.modeling_outputs.CausalLMOutputWithPast]
 |      The [`LlamaForCausalLM`] forward method, overrides the `__call__` special method.
 |
 |      <Tip>
 |
 |      Although the recipe for forward pass needs to be defined within this function, one should call the [`Module`]
 |      instance afterwards instead of this since the former takes care of running the pre and post processing steps while
 |      the latter silently ignores them.
 |
 |      </Tip>
 |
 |      Args:
 |          input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
 |              Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
 |              it.
 |
 |              Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
 |              [`PreTrainedTokenizer.__call__`] for details.
 |
 |              [What are input IDs?](../glossary#input-ids)
 |          attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
 |              Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
 |
 |              - 1 for tokens that are **not masked**,
 |              - 0 for tokens that are **masked**.
 |
 |              [What are attention masks?](../glossary#attention-mask)
...
"""

print(model)

"""
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
)
"""

"""
If you want to get an overview of the attributes of a class, you can use the dir() function.
"""
dir(model)
"""
['T_destination', '__annotations__', '__call__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattr__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_apply', '_assisted_decoding', '_auto_class', '_autoset_attn_implementation', '_backward_compatibility_gradient_checkpointing', '_backward_hooks', '_backward_pre_hooks', '_beam_search', '_buffers', '_call_impl', '_check_and_enable_flash_attn_2', '_check_and_enable_sdpa', '_compiled_call_impl', '_constrained_beam_search', '_contrastive_search', '_convert_head_mask_to_5d', '_copy_lm_head_original_to_resized', '_create_repo', '_dispatch_accelerate_model', '_dola_decoding', '_expand_inputs_for_generation', '_extract_past_from_model_output', '_forward_hooks', '_forward_hooks_always_called', '_forward_hooks_with_kwargs', '_forward_pre_hooks', '_forward_pre_hooks_with_kwargs', '_from_config', '_get_backward_hooks', '_get_backward_pre_hooks', '_get_cache', '_get_candidate_generator', '_get_files_timestamps', '_get_initial_cache_position', '_get_logits_processor', '_get_logits_warper', '_get_name', '_get_no_split_modules', '_get_resized_embeddings', '_get_resized_lm_head', '_get_stopping_criteria', '_group_beam_search', '_has_unfinished_sequences', '_hf_peft_config_loaded', '_hook_rss_memory_post_forward', '_hook_rss_memory_pre_forward', '_init_weights', '_initialize_weights', '_is_full_backward_hook', '_is_hf_initialized', '_is_quantized_training_enabled', '_is_stateful', '_keep_in_fp32_modules', '_keep_in_fp32_modules', '_keys_to_ignore_on_load_missing', '_keys_to_ignore_on_load_unexpected', '_keys_to_ignore_on_save', '_load_from_state_dict', '_load_pretrained_model', '_load_pretrained_model_low_mem', '_load_state_dict_post_hooks', '_load_state_dict_pre_hooks', '_maybe_initialize_input_ids_for_generation', '_maybe_warn_non_full_backward_hook', '_merge_criteria_processor_list', '_modules', '_named_members', '_no_split_modules', '_non_persistent_buffers_set', '_parameters', '_prepare_attention_mask_for_generation', '_prepare_decoder_input_ids_for_generation', '_prepare_encoder_decoder_kwargs_for_generation', '_prepare_generated_length', '_prepare_generation_config', '_prepare_model_inputs', '_prepare_special_tokens', '_register_load_state_dict_pre_hook', '_register_state_dict_hook', '_reorder_cache', '_replicate_for_data_parallel', '_resize_token_embeddings', '_sample', '_save_to_state_dict', '_set_default_torch_dtype', '_set_gradient_checkpointing', '_skip_keys_device_placement', '_slow_forward', '_state_dict_hooks', '_state_dict_pre_hooks', '_supports_cache_class', '_supports_default_dynamic_cache', '_supports_flash_attn_2', '_supports_quantized_cache', '_supports_sdpa', '_supports_static_cache', '_temporary_reorder_cache', '_tie_encoder_decoder_weights', '_tie_or_clone_weights', '_tied_weights_keys', '_transformers_zero3_init_used', '_update_model_kwargs_for_generation', '_upload_modified_files', '_validate_assistant', '_validate_generated_length', '_validate_model_class', '_validate_model_kwargs', '_version', '_wrapped_call_impl', 'active_adapter', 'active_adapters', 'add_adapter', 'add_memory_hooks', 'add_model_tags', 'add_module', 'apply', 'base_model', 'base_model_prefix', 'bfloat16', 'buffers', 'call_super_init', 'can_generate', 'children', 'compile', 'compute_transition_scores', 'config', 'config_class', 'contrastive_search', 'cpu', 'create_extended_attention_mask_for_decoder', 'cuda', 'dequantize', 'device', 'disable_adapters', 'disable_input_require_grads', 'double', 'dtype', 'dummy_inputs', 'dump_patches', 'enable_adapters', 'enable_input_require_grads', 'estimate_tokens', 'eval', 'extra_repr', 'float', 'floating_point_ops', 'forward', 'framework', 'from_pretrained', 'generate', 'generation_config', 'get_adapter_state_dict', 'get_buffer', 'get_decoder', 'get_extended_attention_mask', 'get_extra_state', 'get_head_mask', 'get_input_embeddings', 'get_memory_footprint', 'get_output_embeddings', 'get_parameter', 'get_position_embeddings', 'get_submodule', 'gradient_checkpointing_disable', 'gradient_checkpointing_enable', 'half', 'heal_tokens', 'init_weights', 'invert_attention_mask', 'ipu', 'is_gradient_checkpointing', 'is_parallelizable', 'lm_head', 'load_adapter', 'load_state_dict', 'main_input_name', 'model', 'model_tags', 'modules', 'name_or_path', 'named_buffers', 'named_children', 'named_modules', 'named_parameters', 'num_parameters', 'parameters', 'post_init', 'prepare_inputs_for_generation', 'prune_heads', 'push_to_hub', 'register_backward_hook', 'register_buffer', 'register_for_auto_class', 'register_forward_hook', 'register_forward_pre_hook', 'register_full_backward_hook', 'register_full_backward_pre_hook', 'register_load_state_dict_post_hook', 'register_module', 'register_parameter', 'register_state_dict_pre_hook', 'requires_grad_', 'reset_memory_hooks_state', 'resize_position_embeddings', 'resize_token_embeddings', 'retrieve_modules_from_names', 'reverse_bettertransformer', 'save_pretrained', 'set_adapter', 'set_decoder', 'set_extra_state', 'set_input_embeddings', 'set_output_embeddings', 'share_memory', 'state_dict', 'supports_gradient_checkpointing', 'tie_weights', 'to', 'to_bettertransformer', 'to_empty', 'train', 'training', 'type', 'vocab_size', 'warn_if_padding_and_no_attention_mask', 'warnings_issued', 'xpu', 'zero_grad']"

// The dunder methods (__<>__) are the special methods that are used to perform special operations.
//  e.g. __call__ is called when the object is called as a function.


// The methods starting with "_" are generally private methods and should not be called directly.
// e.g. _parameters is a private attribute which is used in register_parameter method to add a named parameter to the module.

// The remaining methods are generally the public attributes that can be called directly.
"""
"To find name of the class -- can be further used with help() functionality"

type(model)
"<class 'transformers.models.llama.modeling_llama.LlamaForCausalLM'>"

"""
To get the how a function is implemented, one can use the inspect module.
"""
import inspect
print(inspect.getsource(model.__call__))
"""
 def _wrapped_call_impl(self, *args, **kwargs):
        if self._compiled_call_impl is not None:
            return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
        else:
            return self._call_impl(*args, **kwargs)

>>> print(inspect.getsource(model._call_impl))
 def _call_impl(self, *args, **kwargs):                                                                                                                                                                                                   
        forward_call = (self._slow_forward if torch._C._get_tracing_state() else self.forward)                                                                                                                                               
        # If we don't have any hooks, we want to skip the rest of the logic in                                                                                                                                                               
        # this function, and just call forward.                                                                                                                                                                                              
        if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks                                                                                                                           
                or _global_backward_pre_hooks or _global_backward_hooks                                                                                                                                                                      
                or _global_forward_hooks or _global_forward_pre_hooks):                                                                                                                                                                      
            return forward_call(*args, **kwargs)                                                                                                                                                                                             
                                                                                                                                                                                                                                             
        try:                                                                                                                                                                                                                                 
            result = None                                                                                                                                                                                                                    
            called_always_called_hooks = set()                                                                                                                                                                                               
                                                                                                                                                                                                                                             
            full_backward_hooks, non_full_backward_hooks = [], []                                                                                                                                                                            
            backward_pre_hooks = []                                                                                                                                                                                                          
            if self._backward_pre_hooks or _global_backward_pre_hooks:                                                                                                                                                                       
                backward_pre_hooks = self._get_backward_pre_hooks()                                                                                                                                                                          
                                                                                                                                                                                                                                             
            if self._backward_hooks or _global_backward_hooks:                                                                                                                                                                               
                full_backward_hooks, non_full_backward_hooks = self._get_backward_hooks()                                                                                                                                                    
 ...
                         warnings.warn("module forward hook with ``always_call=True`` raised an exception "
                                      f"that was silenced as another error was raised in forward: {str(e)}")
                        continue
            # raise exception raised in try block
            raise
"""
# Digging deeper
print(inspect.getsource(model.forward))


r'''
 @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)                                                                                                                                                                  [44/1874]
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(                            
        self,                           
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,                                                                           
        cache_position: Optional[torch.LongTensor] = None,                                                            
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r""" 
        Args:                                   
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored 
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
                                                           
        Returns:                                                                                                      
                                                           
        Example:                
                                                           
        ```python                                                                                                     
        >>> from transformers import AutoTokenizer, LlamaForCausalLM
                                                           
        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt") 
                                                                                                                      
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""                                      
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (          
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
...
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
// Note that the self.model() calls the forward function of all the parent modules listed in the print(model) output 
// and one can dig deeper to find how Attention and MLP modules are implemented.
// e.g.
import transformers
print(inspect.getsource(transformers.models.llama.modeling_llama.LlamaAttention.forward))
// Hint: print(model.model.layers[0].self_attn.__class__.__bases__)
'''


input_string = "Hey, can you talk to me?"
tokenized_input = tokenizer(input_string, return_attention_mask=True, return_tensors='pt')
print(tokenized_input)

"""
{'input_ids': tensor([[128000,  19182,     11,    649,    499,   3137,    311,    757,     30]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])}
"""
tokenized_input = tokenized_input.to(model.device)
out = model(**tokenized_input)
print(out)

"""
The print output is too long to read and understand with the whole tensors printed on the console.
However, from the code inspection of model.forward method, we can see that the output must be
CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

Let us confirm the hypothesis by checking the type of the output.
>>> type(out)
class 'transformers.modeling_outputs.CausalLMOutputWithPast'

>>> help(out)
Help on CausalLMOutputWithPast in module transformers.modeling_outputs object:

class CausalLMOutputWithPast(transformers.utils.generic.ModelOutput)
 |  CausalLMOutputWithPast(loss: Optional[torch.FloatTensor] = None, logits: torch.FloatTensor = None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None, attentions: Optional[Tuple[torch.FloatTensor, ...]] = None) -> None
 |
 |  Base class for causal language model (or autoregressive) outputs.
 |
 |  Args:
 |      loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
 |          Language modeling loss (for next-token prediction).
 |      logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
 |          Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
 |      past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
 |          Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
 |          `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
 |
 |          Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
 |          `past_key_values` input) to speed up sequential decoding.
 |      hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
 |          Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
 |          one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
 |
 |          Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
 |      attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
 |          Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
 |          sequence_length)`.
 |
 |          Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
 |          heads.
 |
 |  Method resolution order:
 |      CausalLMOutputWithPast
 |      transformers.utils.generic.ModelOutput
 |      collections.OrderedDict
 |      builtins.dict
 |      builtins.object
 |
 |  Methods defined here:
...
"""
print(out.loss)
"None"
print(out.logits)
"""
tensor([[[-2.9512,  1.5942,  7.4129,  ...,  1.6402,  1.6402,  1.6403],
         [11.4474,  2.1147,  1.8264,  ..., -4.7011, -4.7013, -4.7013],
         [-0.5021, -4.9302, -2.0321,  ..., -6.0960, -6.0961, -6.0960],
         ...,
         [ 1.9550, -0.4242, -1.1474,  ..., -7.4085, -7.4085, -7.4086],
         [11.9058,  5.6987,  4.0754,  ..., -0.9628, -0.9628, -0.9629],
         [ 5.6675,  1.0652,  5.3867,  ..., -4.3538, -4.3539, -4.3540]]],
       grad_fn=<UnsafeViewBackward0>)
"""
print(out.logits.shape)
"torch.Size([1, 9, 128256])"

"""
The documentation in help showed an example using the generate method and decoding the output.
Let's check what the generate method does.

>>> print(inspect.getsource(model.generate))
TLDR;
# 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
# 2. Set generation parameters if not already defined
# 3. Define model inputs
# 4. Define other model kwargs
# 5. Prepare `input_ids` which will be used for auto-regressive generation
# 6. Prepare `max_length` depending on other stopping criteria.
# 7. determine generation mode
# 8. prepare distribution pre_processing samplers
# 9. prepare stopping criteria
# 10. go into different generation modes
# 11. prepare logits warper  
# 12. expand input_ids with `num_return_sequences` additional sequences per batch
# 13. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
return result

Question: Which mode is followed by default for generation?
Answer hint:  model.generation_config.get_generation_mode()

>>>print(inspect.getsource(model._sample))
-- further uses LogitNormalization()
"""

"""
Another important function to inspect the internal attributes of a class object
"""
vars(model)