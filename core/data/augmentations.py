"""Data augmentation techniques for handwritten math recognition."""

import tensorflow as tf
import numpy as np


def random_rotation(image, max_angle=5.0):
    """Apply random rotation to the input image.
    
    Args:
        image: Input tensor of shape [height, width, channels]
        max_angle: Maximum rotation angle in degrees
        
    Returns:
        Rotated image tensor
    """
    angle = tf.random.uniform([], -max_angle, max_angle)
    return tf.image.rotate(image, angle * np.pi / 180)


def random_scale(image, scale_range=0.1):
    """Apply random scaling to the input image.
    
    Args:
        image: Input tensor of shape [height, width, channels]
        scale_range: Maximum scale factor deviation from 1.0
        
    Returns:
        Scaled image tensor
    """
    scale = tf.random.uniform([], 1.0 - scale_range, 1.0 + scale_range)
    shape = tf.shape(image)
    new_height = tf.cast(tf.cast(shape[0], tf.float32) * scale, tf.int32)
    new_width = tf.cast(tf.cast(shape[1], tf.float32) * scale, tf.int32)
    return tf.image.resize(image, [new_height, new_width])


def random_translation(image, translation_range=0.1):
    """Apply random translation to the input image.
    
    Args:
        image: Input tensor of shape [height, width, channels]
        translation_range: Maximum translation as fraction of image dimensions
        
    Returns:
        Translated image tensor
    """
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    
    max_dx = tf.cast(width * translation_range, tf.int32)
    max_dy = tf.cast(height * translation_range, tf.int32)
    
    dx = tf.random.uniform([], -max_dx, max_dx, tf.int32)
    dy = tf.random.uniform([], -max_dy, max_dy, tf.int32)
    
    # This is a simplified version - actual implementation would handle borders
    return tf.roll(image, [dy, dx], [0, 1])


def random_noise(image, noise_stddev=0.05):
    """Add random noise to the input image.
    
    Args:
        image: Input tensor of shape [height, width, channels]
        noise_stddev: Standard deviation of the noise
        
    Returns:
        Noisy image tensor
    """
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_stddev)
    return tf.clip_by_value(image + noise, 0.0, 1.0)


def elastic_deformation(image, alpha=10, sigma=3):
    """Apply elastic deformation to the input image.
    
    This is particularly useful for handwriting recognition as it simulates
    the variations in handwriting styles.
    
    Args:
        image: Input tensor of shape [height, width, channels]
        alpha: Scaling factor for deformation
        sigma: Smoothing factor for deformation field
        
    Returns:
        Deformed image tensor
    """
    # Placeholder - this would be implemented with proper elastic deformation
    # Similar to what's used in MNIST augmentation
    return image


class AugmentationPipeline:
    """Pipeline for applying multiple augmentations with configurable probability."""
    
    def __init__(self, rotation_range=5.0, scale_range=0.1, translation_range=0.1):
        """Initialize the augmentation pipeline.
        
        Args:
            rotation_range: Maximum rotation angle in degrees
            scale_range: Maximum scale factor deviation from 1.0
            translation_range: Maximum translation as fraction of image dimensions
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
    
    def apply_augmentations(self, image, p=0.5):
        """Apply a random combination of augmentations with probability p.
        
        Args:
            image: Input tensor of shape [height, width, channels]
            p: Probability of applying each augmentation
            
        Returns:
            Augmented image tensor
        """
        augmentations = [
            lambda img: random_rotation(img, self.rotation_range),
            lambda img: random_scale(img, self.scale_range),
            lambda img: random_translation(img, self.translation_range),
            random_noise,
            elastic_deformation
        ]
        
        # Apply each augmentation with probability p
        for aug_fn in augmentations:
            if tf.random.uniform([]) < p:
                image = aug_fn(image)
                
        return image