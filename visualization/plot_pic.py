from draw_roc import draw_roc_func
from t import t_func
from validation_loss import validation_loss_func
from inference_time import draw_circle_func


if __name__=='__main__':
    datasets=['TN3K', 'DDTI']
    model1_list=['trfeplus']
    model2_list=['trfe', 'unet', 'R50-ViT-B_16', 'segnet', 'mtnet', 'fcn', 'deeplab-resnet50', 'deeplab-v3', 'CPFNet']
    # # t检验
    # for dataset in datasets:
    #     print(dataset,':')
    #     for model1 in model1_list:
    #         for model2 in model2_list:
    #             t_func(dataset, model1, model2)

    #     for model3 in model3_list:
    #         for model4 in model3_list:
    #             t_func(dataset, model3, model4)


    # roc曲线
    for dataset in datasets:
        for fold in range(5):
            draw_roc_func(dataset, fold, from_scratch=args.from_scratch)

    # # 学习曲线
    # validation_loss_func(model1_list)

    # # inference time
    # models_list = ['unet', 'sgunet', 'trfe', 'fcn', 'segnet', 'deeplab-v3', 'cenet', 'cpfnet', 'R50-ViT-B_16']
    # # models_list = ['unet', 'sgunet', 'fcn', 'segnet', 'cenet', 'deeplab-v3', 'cpfnet', 'R50-ViT-B_16']
    # datasets = ['TN3K', 'DDTI']
    # draw_circle_func(models_list, datasets)
