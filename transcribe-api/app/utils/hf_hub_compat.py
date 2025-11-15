"""Compatibility shim for huggingface_hub to handle use_auth_token parameter.

pyannote.audio uses the old 'use_auth_token' parameter which was removed
in newer versions of huggingface_hub. This patches hf_hub_download to accept it.

CRITICAL: This module MUST be imported before pyannote.audio imports huggingface_hub.
"""

import sys
import functools

# Try to import and patch immediately
try:
    import huggingface_hub
    
    # Patch hf_hub_download
    original_hf_hub_download = huggingface_hub.hf_hub_download
    
    if not hasattr(original_hf_hub_download, '_hf_compat_patched'):
        @functools.wraps(original_hf_hub_download)
        def patched_hf_hub_download(*args, **kwargs):
            """Convert use_auth_token to token."""
            if 'use_auth_token' in kwargs:
                token_val = kwargs.pop('use_auth_token')
                if 'token' not in kwargs:
                    kwargs['token'] = token_val
            return original_hf_hub_download(*args, **kwargs)
        
        patched_hf_hub_download._hf_compat_patched = True
        huggingface_hub.hf_hub_download = patched_hf_hub_download
        sys.modules['huggingface_hub'].hf_hub_download = patched_hf_hub_download
    
    # Patch validator
    try:
        from huggingface_hub.utils import _validators
        original_validator = _validators._inner_fn
        
        if not hasattr(original_validator, '_hf_compat_patched'):
            @functools.wraps(original_validator)
            def patched_validator(fn, *args, **kwargs):
                """Convert use_auth_token to token in validator."""
                if 'use_auth_token' in kwargs:
                    token_val = kwargs.pop('use_auth_token')
                    if 'token' not in kwargs:
                        kwargs['token'] = token_val
                return original_validator(fn, *args, **kwargs)
            
            patched_validator._hf_compat_patched = True
            _validators._inner_fn = patched_validator
    except (ImportError, AttributeError):
        pass
    
    # Also patch pyannote.audio's get_model if it's available
    try:
        from pyannote.audio.pipelines.utils.getter import get_model
        original_get_model = get_model
        
        if not hasattr(original_get_model, '_hf_compat_patched'):
            @functools.wraps(original_get_model)
            def patched_get_model(*args, **kwargs):
                """Convert use_auth_token to token in pyannote's get_model."""
                if 'use_auth_token' in kwargs:
                    token_val = kwargs.pop('use_auth_token')
                    if 'token' not in kwargs:
                        kwargs['token'] = token_val
                return original_get_model(*args, **kwargs)
            
            patched_get_model._hf_compat_patched = True
            
            # Replace in the module
            import pyannote.audio.pipelines.utils.getter as getter_module
            getter_module.get_model = patched_get_model
            sys.modules['pyannote.audio.pipelines.utils.getter'].get_model = patched_get_model
    except (ImportError, AttributeError):
        # pyannote.audio not imported yet or get_model doesn't exist
        pass
        
except ImportError:
    # huggingface_hub not available yet - that's ok
    pass
