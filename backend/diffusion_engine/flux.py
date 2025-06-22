import torch
import os
from pathlib import Path

from huggingface_guess import model_list
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.vae import VAE
from backend.patcher.unet import UnetPatcher
from backend.text_processing.classic_engine import ClassicTextProcessingEngine
from backend.text_processing.t5_engine import T5TextProcessingEngine
from backend.args import dynamic_args
from backend.modules.k_prediction import PredictionFlux
from backend import memory_management

# Import GGUF support - optional dependency
try:
    import gguf
    from .dequant import dequantize_tensor, is_quantized
    from .ops import GGMLTensor
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    print("Info: GGUF support not available. Install 'gguf' package for GGUF T5 support.")


def gguf_clip_loader(path):
    """
    Load GGUF T5 text encoder (adapted from ComfyUI-GGUF)
    """
    if not GGUF_AVAILABLE:
        raise ImportError("GGUF package required for loading GGUF files")
    
    from .loader import gguf_clip_loader as _gguf_clip_loader
    return _gguf_clip_loader(path)


class UnifiedT5TextProcessingEngine:
    """
    Unified text processing engine that can handle both regular T5 and GGUF T5
    """
    def __init__(self, text_encoder=None, tokenizer=None, gguf_path=None, emphasis_name=None):
        self.emphasis_name = emphasis_name
        self.is_gguf = gguf_path is not None
        
        if self.is_gguf:
            if not GGUF_AVAILABLE:
                raise ImportError("GGUF support not available")
            
            # Load GGUF model
            self.gguf_path = gguf_path
            # Initialize GGUF model loading here - this would need ComfyUI-GGUF integration
            print(f"Loading GGUF T5 from: {gguf_path}")
            
            # For now, we'll use a placeholder - you'd need to integrate ComfyUI-GGUF properly
            self.gguf_state_dict = None
            try:
                self.gguf_state_dict = gguf_clip_loader(gguf_path)
            except Exception as e:
                print(f"Failed to load GGUF T5: {e}")
                raise
        else:
            # Use regular T5 text encoder
            self.text_encoder = text_encoder
            self.tokenizer = tokenizer
    
    def tokenize(self, texts):
        """Tokenize input texts"""
        if self.is_gguf:
            # For GGUF, we need to implement tokenization
            # This would require proper integration with ComfyUI-GGUF tokenizer
            if isinstance(texts, str):
                texts = [texts]
            
            # Placeholder - you'd need proper GGUF tokenizer integration
            tokenized = []
            for text in texts:
                # This is a simplified tokenization - you'd need proper implementation
                tokens = text.encode('utf-8')  # Placeholder
                tokenized.append(tokens)
            return tokenized
        else:
            # Use regular tokenizer
            return self.tokenizer(texts)
    
    def __call__(self, prompts):
        """
        Process prompts and return embeddings
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if self.is_gguf:
            # For GGUF T5, we need to process through GGUF model
            # This is where you'd integrate with ComfyUI-GGUF's text encoding
            embeddings = []
            for prompt in prompts:
                # Apply emphasis processing if needed
                if self.emphasis_name:
                    # You'd implement emphasis processing here
                    pass
                
                # Process through GGUF model (placeholder)
                # You'd need to implement actual GGUF T5 inference here
                embedding = torch.randn(1, 4096)  # Placeholder
                embeddings.append(embedding)
            
            if embeddings:
                return torch.stack(embeddings)
            else:
                return torch.empty(0, dtype=torch.float32)
        else:
            # Use regular T5 processing
            return self.text_encoder(prompts)


class Flux(ForgeDiffusionEngine):
    matched_guesses = [model_list.Flux, model_list.FluxSchnell]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        self.is_inpaint = False
        
        # Check for GGUF T5 configuration
        self.use_gguf_t5 = self._should_use_gguf_t5(estimated_config)
        
        clip = CLIP(
            model_dict={
                'clip_l': huggingface_components['text_encoder'],
                't5xxl': huggingface_components['text_encoder_2'] if not self.use_gguf_t5 else None
            },
            tokenizer_dict={
                'clip_l': huggingface_components['tokenizer'],
                't5xxl': huggingface_components['tokenizer_2'] if not self.use_gguf_t5 else None
            }
        )

        vae = VAE(model=huggingface_components['vae'])

        if 'schnell' in estimated_config.huggingface_repo.lower():
            k_predictor = PredictionFlux(
                mu=1.0
            )
        else:
            k_predictor = PredictionFlux(
                seq_len=4096,
                base_seq_len=256,
                max_seq_len=4096,
                base_shift=0.5,
                max_shift=1.15,
            )
            self.use_distilled_cfg_scale = True

        unet = UnetPatcher.from_model(
            model=huggingface_components['transformer'],
            diffusers_scheduler=None,
            k_predictor=k_predictor,
            config=estimated_config
        )

        self.text_processing_engine_l = ClassicTextProcessingEngine(
            text_encoder=clip.cond_stage_model.clip_l,
            tokenizer=clip.tokenizer.clip_l,
            embedding_dir=dynamic_args['embedding_dir'],
            embedding_key='clip_l',
            embedding_expected_shape=768,
            emphasis_name=dynamic_args['emphasis_name'],
            text_projection=False,
            minimal_clip_skip=1,
            clip_skip=1,
            return_pooled=True,
            final_layer_norm=True,
        )

        # Initialize T5 text processing engine (GGUF or regular)
        if self.use_gguf_t5:
            gguf_model_path = self._get_gguf_t5_path(estimated_config)
            if gguf_model_path and os.path.exists(gguf_model_path):
                self.text_processing_engine_t5 = GGUFT5TextProcessingEngine(
                    model_path=gguf_model_path,
                    emphasis_name=dynamic_args['emphasis_name'],
                )
                print(f"Using GGUF T5 model: {gguf_model_path}")
            else:
                print("GGUF T5 model not found, falling back to regular T5")
                self.use_gguf_t5 = False
                self.text_processing_engine_t5 = T5TextProcessingEngine(
                    text_encoder=clip.cond_stage_model.t5xxl,
                    tokenizer=clip.tokenizer.t5xxl,
                    emphasis_name=dynamic_args['emphasis_name'],
                )
        else:
            self.text_processing_engine_t5 = T5TextProcessingEngine(
                text_encoder=clip.cond_stage_model.t5xxl,
                tokenizer=clip.tokenizer.t5xxl,
                emphasis_name=dynamic_args['emphasis_name'],
            )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

    def _should_use_gguf_t5(self, estimated_config):
        """
        Determine if GGUF T5 should be used based on configuration or environment variables
        """
        # Check environment variable
        if os.getenv('USE_GGUF_T5', '').lower() in ['true', '1', 'yes']:
            return GGUF_AVAILABLE
        
        # Check if GGUF path is specified in config
        if hasattr(estimated_config, 'gguf_t5_path') and estimated_config.gguf_t5_path:
            return GGUF_AVAILABLE
        
        # Auto-detect GGUF file in models directory
        gguf_path = self._get_gguf_t5_path(estimated_config)
        return GGUF_AVAILABLE and gguf_path and os.path.exists(gguf_path)

    def _get_gguf_t5_path(self, estimated_config):
        """
        Get the path to GGUF T5 model file
        """
        # Check if explicitly specified in config
        if hasattr(estimated_config, 'gguf_t5_path'):
            return estimated_config.gguf_t5_path
        
        # Check environment variable
        env_path = os.getenv('GGUF_T5_PATH')
        if env_path:
            return env_path
        
        # Auto-detect in common locations
        possible_paths = [
            # ComfyUI style paths
            'models/clip/t5-v1_1-xxl-encoder-Q5_K_M.gguf',
            'models/clip/t5-v1_1-xxl-encoder-Q4_K_M.gguf',
            'models/clip/t5-v1_1-xxl-encoder-Q8_0.gguf',
            # Alternative paths
            'models/t5/t5-v1_1-xxl-encoder-Q5_K_M.gguf',
            'models/t5/t5-v1_1-xxl-encoder-Q4_K_M.gguf',
            'models/t5/t5-v1_1-xxl-encoder-Q8_0.gguf',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None

    def set_clip_skip(self, clip_skip):
        self.text_processing_engine_l.clip_skip = clip_skip

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        if not self.use_gguf_t5:
            memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        
        cond_l, pooled_l = self.text_processing_engine_l(prompt)
        
        # Handle GGUF T5 vs regular T5
        if self.use_gguf_t5:
            # For GGUF T5, we need to handle the processing differently
            cond_t5 = self.text_processing_engine_t5(prompt)
            # Ensure proper tensor format and device
            if cond_t5.device != cond_l.device:
                cond_t5 = cond_t5.to(cond_l.device)
        else:
            cond_t5 = self.text_processing_engine_t5(prompt)
        
        cond = dict(crossattn=cond_t5, vector=pooled_l)

        if self.use_distilled_cfg_scale:
            distilled_cfg_scale = getattr(prompt, 'distilled_cfg_scale', 3.5) or 3.5
            cond['guidance'] = torch.FloatTensor([distilled_cfg_scale] * len(prompt))
            print(f'Distilled CFG Scale: {distilled_cfg_scale}')
        else:
            print('Distilled CFG Scale will be ignored for Schnell')

        return cond

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        if self.use_gguf_t5:
            # For GGUF models, use the tokenizer from the GGUF engine
            token_count = len(self.text_processing_engine_t5.tokenize([prompt])[0])
        else:
            token_count = len(self.text_processing_engine_t5.tokenize([prompt])[0])
        return token_count, max(255, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        sample = self.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = self.forge_objects.vae.first_stage_model.process_in(sample)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        sample = self.forge_objects.vae.first_stage_model.process_out(x)
        sample = self.forge_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)
