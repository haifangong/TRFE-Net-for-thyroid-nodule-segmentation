python eval.py -model_name trfeplus -load_path "./run/trfe-plus/fold0/trfe-plus_best.pth" -fold 0 -gpu 0
python eval.py -model_name trfeplus -load_path "./run/trfe-plus/fold1/trfe-plus_best.pth" -fold 1 -gpu 0
python eval.py -model_name trfeplus -load_path "./run/trfe-plus/fold2/trfe-plus_best.pth" -fold 2 -gpu 0
python eval.py -model_name trfeplus -load_path "./run/trfe-plus/fold3/trfe-plus_best.pth" -fold 3 -gpu 0
python eval.py -model_name trfeplus -load_path "./run/trfe-plus/fold4/trfe-plus_best.pth" -fold 4 -gpu 0

python eval.py -model_name trfeplus -load_path "./run/trfe-plus/fold0/trfe-plus_best.pth" -fold 0 -gpu 0 -test_dataset DDTI
python eval.py -model_name trfeplus -load_path "./run/trfe-plus/fold1/trfe-plus_best.pth" -fold 1 -gpu 0 -test_dataset DDTI
python eval.py -model_name trfeplus -load_path "./run/trfe-plus/fold2/trfe-plus_best.pth" -fold 2 -gpu 0 -test_dataset DDTI
python eval.py -model_name trfeplus -load_path "./run/trfe-plus/fold3/trfe-plus_best.pth" -fold 3 -gpu 0 -test_dataset DDTI
python eval.py -model_name trfeplus -load_path "./run/trfe-plus/fold4/trfe-plus_best.pth" -fold 4 -gpu 0 -test_dataset DDTI
