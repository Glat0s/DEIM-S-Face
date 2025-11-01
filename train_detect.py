#!/usr/bin/env python3
"""
Training script for a custom single-class face detection model using DEIMKit,
optimized for a high-VRAM GPU (e.g., H200) and high-resolution images.

This script configures and trains a 'deim_hgnetv2_s' model from scratch on a
custom COCO-formatted dataset for face detection.
"""

from deimkit import Trainer, Config, configure_dataset, configure_model
from loguru import logger

def main():
    """Main function to configure and run the training process."""
    
    # --- 1. Select a model architecture and get its base configuration. ---
    # 'deim_hgnetv2_s' is a good balance. With 144GB VRAM, you could also
    # experiment with 'deim_hgnetv2_m' or 'deim_hgnetv2_l' for higher accuracy.
    logger.info("Loading base configuration for 'deim_hgnetv2_s'...")
    conf = Config.from_model_name("deim_hgnetv2_s")

    # --- 2. Configure the model for training from scratch. ---
    # Setting `pretrained=False` initializes the model with random weights.
    logger.info("Configuring model to train from scratch (pretrained=False)...")
    conf = configure_model(
        config=conf,
        pretrained=False,   # Set to False to train from scratch.
    )

    # --- 3. Configure the dataset paths, sizes, and other parameters. ---
    # This section is optimized for your high-resolution data and powerful GPU.
    logger.info("Configuring dataset parameters...")
    conf = configure_dataset(
        config=conf,
        # Set to 1024x1024. This will resize the longest edge to 1024
        # and pad the shorter edge, preserving the aspect ratio.
        image_size=(1024, 1024),
        
        # --- UPDATE THESE PATHS ---
        train_ann_file="path/to/your/coco/train/_annotations.coco.json",
        train_img_folder="path/to/your/coco/train",
        val_ann_file="path/to/your/coco/valid/_annotations.coco.json",
        val_img_folder="path/to/your/coco/valid",
        # --------------------------

        # With 144GB VRAM, a large batch size is possible.
        # This can lead to more stable training. Feel free to increase this further.
        train_batch_size=48,
        val_batch_size=48,

        # For single-class face detection: 1 (face) + 1 (background) = 2.
        num_classes=2,
        
        # Set a descriptive output directory.
        output_dir="./outputs/deim_hgnetv2_s_face_detection_1024_scratch",
    )

    # --- 4. Initialize the Trainer with the configured settings. ---
    logger.info("Initializing trainer...")
    trainer = Trainer(conf)
    
    # --- 5. Start the training process. ---
    # Increased epochs because we are training from scratch on a large dataset.
    logger.info("Starting training for 150 epochs...")
    try:
        trainer.fit(
            epochs=150,
            save_best_only=True, # Saves only the best model based on validation mAP.
            lr=0.0001,
            weight_decay=0.0001,
        )
        logger.success("Training finished successfully!")
        logger.info(f"Best model and logs saved in: {conf.output_dir}")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        logger.exception(e) # This will print the full traceback for debugging.

if __name__ == "__main__":
    main()