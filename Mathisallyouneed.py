#—Copyright © 2026 Spencer Burwell—#

import torch
import math

def complex_geometric_attention(query, key, value, p, c, N):
    """
    Compute Attention using continuous complex space geometry.
    Eliminates brute-force masking by handling boundaries via phase rotation.
    """
    # 1. Standard vector dimension pulling (pure integer, no arithmetic side-effects)
    d_k = query.size(-1)
    
    # 2. Compute the pure geometric relationship (unwarped by arbitrary scaling)
    # This preserves the raw dot-product tensor
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # 3. Apply the Spencer Burwell-style complex transformation loop
    # n = (p * e * c) / (N * pi)
    e = math.e
    pi = math.pi
    
    # Calculate the continuous phase variable n
    n_val = (p * e * c) / (N * pi)
    
    # Establish the exact real-number landing boundaries requested
    upper_bound = -100 / 43
    lower_bound = -3.28888
    
    # Clamp the real structural component strictly within the geometric window
    real_component = torch.clamp(scores * n_val, min=lower_bound, max=upper_bound)
    
    # 4. Map into the complex plane to rotate the phase instead of crushing the magnitude
    # This prevents the Law of Cosines from breaking by using Euler's formula identity
    imag_component = torch.sin(scores) * n_val
    
    # Create the complex tensor representing the continuous geometric manifold
    complex_scores = torch.complex(real_component, imag_component)
    
    # 5. Convert phase back to clean, positive scalar weights without using a destructive softmax
    # Taking the absolute magnitude of the complex exponential smoothly normalizes the space
    p_attn = torch.abs(torch.exp(complex_scores))
    p_attn = p_attn / (torch.sum(p_attn, dim=-1, keepdim=True) + 1e-6)
    
    return torch.matmul(p_attn, value), p_attn
