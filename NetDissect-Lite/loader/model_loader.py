import settings
import torch
import torchvision
import os

def loadmodel(hook_fn):
    if settings.MODEL_FILE is None:
        model = torchvision.models.__dict__[settings.MODEL](pretrained=True)
    else:
        # Check if checkpoint file exists
        if not os.path.exists(settings.MODEL_FILE):
            raise FileNotFoundError(
                f"Checkpoint file not found: {settings.MODEL_FILE}\n"
                f"Please verify the path is correct."
            )
        
        print(f"Loading checkpoint from: {settings.MODEL_FILE}")
        checkpoint = torch.load(settings.MODEL_FILE, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, (dict, type(torch._C.OrderedDict()))):
            model = torchvision.models.__dict__[settings.MODEL](num_classes=settings.NUM_CLASSES)
            
            # Check if checkpoint contains 'state_dict' key (common PyTorch training format)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("Found 'state_dict' key in checkpoint")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                print("Found 'model' key in checkpoint")
            else:
                # Assume checkpoint is the state_dict itself
                state_dict = checkpoint
                print("Using checkpoint directly as state_dict")
            
            # Handle DataParallel prefix removal if needed
            if settings.MODEL_PARALLEL or any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                print("Removed 'module.' prefix from state_dict keys")
            
            # Load state dict with error handling
            try:
                model.load_state_dict(state_dict, strict=False)
                print("Checkpoint loaded successfully (strict=False)")
            except Exception as e:
                print(f"Warning: Error loading state_dict: {e}")
                print("Attempting to load with strict=False...")
                model.load_state_dict(state_dict, strict=False)
        else:
            # Checkpoint is the model itself
            model = checkpoint
            print("Using checkpoint as model directly")
    
    # Register hooks for feature extraction
    for name in settings.FEATURE_NAMES:
        model._modules.get(name).register_forward_hook(hook_fn)
    
    # Move to GPU if specified
    if settings.GPU:
        model.cuda()
        print(f"Model moved to GPU: {torch.cuda.get_device_name(0)}")
    
    model.eval()
    return model
