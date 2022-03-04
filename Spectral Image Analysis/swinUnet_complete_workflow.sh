echo "Training swinUnet model"
python train.py
echo "Training finished."
echo "Generating segmentation masks."
python generate_image.py
echo "Images generated."
echo "Evaluating scores"
python eval.py
