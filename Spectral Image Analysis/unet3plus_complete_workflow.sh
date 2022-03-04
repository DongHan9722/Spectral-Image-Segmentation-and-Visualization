echo "Training unet3plus model"
python train.py --model 'unet3plus' 
echo "Training finished."
echo "Generating segmentation masks."
python generate_image.py --model 'unet3plus'
echo "Images generated."
echo "Evaluating scores"
python eval.py --model 'unet3plus'
