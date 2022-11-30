import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from sklearn.metrics import roc_curve, auc


def get_pred_paths(model_name: str, dataset: str, fold: int):
	root = f'./results/test-{dataset}/{model_name}/fold{fold}'
	return [os.path.join(root, x) for x in os.listdir(root) if not 's' in x]


def get_iou(pred, gt):
    # gt = (gt >= 0.5)
    # pred = (pred >= 0.5)
    # fpr, tpr, threshold = roc_curve(gt.flatten(), pred.flatten())
    # auc_roc = auc(fpr, tpr)
    # return auc_roc
	ins = (pred & gt).astype('float32').sum()
	union = (pred | gt).astype('float32').sum()
	return ins / union


def read_img(path):
	img = Image.open(path).convert('L')
	img = np.array(img)
	return img


def cal_iou(model_name: str, dataset:str, gt_dict:dict):
	if os.path.exists('results/t/'+dataset+model_name+'.csv'):
		df = pd.read_csv('results/t/'+dataset+model_name+'.csv', index_col=0)
	else:
		iou_dict = {}
		for fold in range(5):
			iou_list = []
			pred_paths = get_pred_paths(model_name, dataset, fold)
			for path in pred_paths:
				pred = read_img(path)
				gt = gt_dict[os.path.basename(path)]
				iou = get_iou(pred, gt)
				iou_list.append(iou)
			iou_dict[f'fold{fold}'] = iou_list
		# print(iou_dict.keys())
		# print([len(iou_dict[key]) for key in iou_dict.keys()])
		df = pd.DataFrame(iou_dict)
		df.to_csv('results/t/'+dataset+model_name+'.csv')
	# print(df.describe())
	return df


def get_gt_dict(dataset: str):
	gt_dict = {}
	if dataset == 'TN3K':
		data_dir = "/home/liguanbin/TRFE-Net-for-thyroid-nodule-segmentation-main/data/tn3k/test-mask/"
	elif dataset == 'DDTI':
		data_dir = "/home/liguanbin/TRFE-Net-for-thyroid-nodule-segmentation-main/data/DDTI/mask/"
	else:
		raise NotImplementedError
	img_paths = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
	for idx, path in enumerate(img_paths):
		gt_dict[os.path.basename(img_paths[idx])] = read_img(path)
	return gt_dict


def t_func(dataset: str, model1:str, model2:str):
	models_name = [model1, model2]
	gt_dict = get_gt_dict(dataset)
	df1 = cal_iou(model1, dataset, gt_dict)
	df2 = cal_iou(model2, dataset, gt_dict)
	t = []
	df1_list = []
	df2_list = []
	for fold in range(5):
		df1[f'fold{fold}'] = [i*2/(i+1) for i in df1[f'fold{fold}']]
		df2[f'fold{fold}'] = [i*2/(i+1) for i in df2[f'fold{fold}']]

		df1_list.extend(df1[f'fold{fold}'])
		df2_list.extend(df2[f'fold{fold}'])
		# t.append(stats.ttest_ind(df1[f'fold{fold}'], df2[f'fold{fold}'])[1])
	p_mean = stats.ttest_ind(df1_list, df2_list)[1]
	# p_mean = np.mean(t)
	if p_mean < 0.0001:
		print(f'{models_name} pvalue: <0.0001')
	elif p_mean < 0.001:
		print(f'{models_name} pvalue: <0.001')
	elif p_mean < 0.01:
		print(f'{models_name} pvalue: <0.01')
	elif p_mean < 0.05:
		print(f'{models_name} pvalue: <0.05')
	else:
		print(f'{models_name} pvalue: {p_mean}')


def main():
	datasets=['TN3K', 'DDTI']
	model1_list=['trfesw']
	model2_list=['trfe', 'unet', 'cenet', 'R50-ViT-B_16', 'segnet', 'mtnet', 'fcn', 'deeplab-resnet50', 'cpfnet']
	for dataset in datasets:
		print(dataset,':')
		for model1 in model1_list:
			for model2 in model2_list:
				t_func(dataset, model1, model2)


if __name__=='__main__':
	main()
