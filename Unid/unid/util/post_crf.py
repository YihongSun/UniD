import numpy as np

def apply_crf(image_rgb, softmax_probs):
    """
    image_rgb: H x W x 3, uint8
    softmax_probs: n_classes x H x W, float32 (your segmentation probabilities)
    """
    
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError as exc:
        raise ImportError(
            "CRF post-processing requires pydensecrf. Install it separately, or omit --post_crf."
        ) from exc

    H, W = image_rgb.shape[:2]
    n_classes = softmax_probs.shape[0]
    d = dcrf.DenseCRF2D(W, H, n_classes)
    
    # Unary: derived from your softmax scores
    unary = unary_from_softmax(softmax_probs)
    d.setUnaryEnergy(unary)
    
    # Pairwise Gaussian: encourages nearby pixels to have same label
    d.addPairwiseGaussian(
        sxy=3,        # spatial std — tighter = less smoothing
        compat=3      # how strongly to enforce
    )
    
    # Pairwise Bilateral: respects edges (penalizes label diff only for similar colors)
    d.addPairwiseBilateral(
        sxy=40,       # spatial std
        srgb=13,      # color std — lower = more edge-preserving
        rgbim=image_rgb,
        compat=10
    )
    
    # Inference
    Q = d.inference(5)  # 5 iterations usually enough
    return np.array(Q).reshape(n_classes, H, W)
