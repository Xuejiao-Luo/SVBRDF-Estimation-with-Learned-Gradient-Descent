from utils.utils import output_process
from utils.renderer import *
from utils.rendererMG import renderTex, renderTex_loss

EPS = 1e-12

def rrmse(a, b):
    '''relative root mean squared error, a: ground truth, b: pred'''
    num =torch.sum(torch.square(a - b))
    den = torch.sum(torch.square(b))
    squared_error = num / den
    return torch.sqrt(squared_error)

def rse(a, b):
    '''relative squared error, a: ground truth, b: pred'''
    true_mean = torch.mean(a)
    squared_error_num = torch.sum(torch.square(a - b))
    squared_error_den = torch.sum(torch.square(a - true_mean))
    return squared_error_num / squared_error_den

def rmse(a, b):
    '''a, b: pytorch tensor array'''
    return torch.sqrt(torch.sum(torch.square(a - b))/torch.numel(a))

def mse(a, b):
    '''a, b: pytorch tensor array'''
    return torch.sum(torch.square(a - b))/torch.numel(a)

def mae(a, b):
    '''a, b: pytorch tensor array'''
    return torch.sum(torch.abs(a - b))/torch.numel(a)

def loss_l2(target, predict):
    w1 = 1.0
    w2 = 1.0
    w3 = 1.0
    w4 = 1.0
    predict = torch.permute(predict, (0,3,1,2)) 
    predictedNormals = predict[:,0:3,:,:]
    predictedDiffuse = predict[:,3:6,:,:]
    predictedRoughness = predict[:,6:9,:,:]
    predictedSpecular = predict[:,9:12,:,:]
    
    normall2 = rmse(predictedNormals, target['normals'])
    diffusel2 = rmse(predictedDiffuse, target['diffuse'])
    roughnessl2 = rmse(predictedRoughness, target['roughness'])
    specularl2 = rmse(predictedSpecular, target['specular'])
        
    loss_l2 = normall2 *w1 + diffusel2 *w2 + roughnessl2 *w3 + specularl2 *w4
    return loss_l2

def loss_l1(target, predict):
    w1 = 1.0
    w2 = 1.0
    w3 = 1.0
    w4 = 1.0
    
    predict = torch.permute(predict, (0,3,1,2)) 
    predictedNormals = predict[:,0:3,:,:]
    predictedDiffuse = predict[:,3:6,:,:]
    predictedRoughness = predict[:,6:9,:,:]
    predictedSpecular = predict[:,9:12,:,:]
    
    normall1 = mae(predictedNormals, target['normals'])
    diffusel1 = mae(predictedDiffuse, target['diffuse'])
    roughnessl1 = mae(predictedRoughness, target['roughness'])
    specularl1 = mae(predictedSpecular, target['specular'])
        
    loss_l1 = normall1 *w1 + diffusel1 *w2 + roughnessl1 *w3 + specularl1 *w4
    return loss_l1

def rendering_loss(targets, outputs, device, args):
    
    CROP_SIZE = targets.size(1)
    renderedDiffuseImages = []
    renderedDiffuseImagesOutputs = []
    renderedSpecularImages = []
    renderedSpecularImagesOutputs = []
    
    surfaceArray=[]
    XsurfaceArray = torch.unsqueeze(torch.linspace(-1.0, 1.0, CROP_SIZE), dim=-1).to(device)
    XsurfaceArray = torch.tile(XsurfaceArray,(1,CROP_SIZE))
    YsurfaceArray = -1 * torch.transpose(XsurfaceArray, 0, 1) #put -1 in the bottom of the table
    XsurfaceArray = torch.unsqueeze(XsurfaceArray, dim = -1)
    YsurfaceArray = torch.unsqueeze(YsurfaceArray, dim = -1)

    surfaceArray = torch.cat([XsurfaceArray, YsurfaceArray, torch.zeros([CROP_SIZE, CROP_SIZE,1], dtype=torch.float32).to(device)], dim=-1)
    surfaceArray = torch.unsqueeze(surfaceArray, dim = 0) #Add dimension to support batch size
    
    # diffuse model rendering
    for nbDiffuseRender in range(args.nbDiffuseRendering):
        diffuses = diffuseRendering(targets.shape[0], targets, outputs, device, args)
        renderedDiffuseImages.append(diffuses[0][0])
        renderedDiffuseImagesOutputs.append(diffuses[1][0])
                              
    # specular model rendering
    for nbspecularRender in range(args.nbSpecularRendering):
        # speculars = specularRendering(args.batch_size, surfaceArray, targets, outputs, device, args)
        speculars = specularRendering(targets.shape[0], surfaceArray, targets, outputs, device, args)
        renderedSpecularImages.append(speculars[0][0])
        renderedSpecularImagesOutputs.append(speculars[1][0]) 
            
    # renderedDiffuseImages contains X (3 by default) renderings of shape [batch_size, 256,256,3]
    rerenderedImg = renderedDiffuseImages[0]
    for renderingDiff in renderedDiffuseImages[1:]:
        rerenderedImg = torch.cat([rerenderedImg, renderingDiff], dim = -1)
    for renderingSpecu in renderedSpecularImages:
        rerenderedImg = torch.cat([rerenderedImg, renderingSpecu], dim = -1)
        
    # renderedDiffuseImages contains X (3 by default) renderings of shape [batch_size, 256,256,3]
    rerenderedOutputs = renderedDiffuseImagesOutputs[0]
    for renderingOutDiff in renderedDiffuseImagesOutputs[1:]:
        rerenderedOutputs = torch.cat([rerenderedOutputs, renderingOutDiff], dim = -1)
    for renderingOutSpecu in renderedSpecularImagesOutputs:
        rerenderedOutputs = torch.cat([rerenderedOutputs, renderingOutSpecu], dim = -1)               
        
    assert args.loss in ("rendering_loss_l1" , "rendering_loss_l2"), "Wrong loss specified for the rendering loss!"
    if args.loss == "rendering_loss_l1":
        rerenderedImg_vis = torch.permute(rerenderedImg, (0, 3, 1, 2))
        rerenderedOutputs_vis = torch.permute(rerenderedOutputs, (0, 3, 1, 2))
        return [rerenderedImg_vis, rerenderedOutputs_vis, mae(torch.log(rerenderedImg+0.01), torch.log(rerenderedOutputs+0.01))]
    elif args.loss == "rendering_loss_l2":
        return [rerenderedImg, rerenderedOutputs, 20*rmse(targets, outputs) + rmse(rerenderedImg, rerenderedOutputs)]


def reproduce_loss(images, predict, lp, cp, L, args):
    reproducedImg = renderTex(predict, lp, cp, L)

    return [reproducedImg, mse(torch.log(images + 0.01), torch.log(reproducedImg + 0.01))]


def loss_func(target, predict, images, args):
    assert args.loss in ("l1", "l2", "rendering_loss_l1" , "rendering_loss_l2"), "The specified loss is unrecognizable!"
    target = target[:, 0:9, :, :]
    predict = predict[:, 0:9, :, :]

    if args.loss == "l1":
        return loss_l1(target, predict)
    elif args.loss == "l2":
        return loss_l2(target, predict)
    elif args.loss == "rendering_loss_l1" or "rendering_loss_l2":

        lp = torch.tensor([0, 0, 250], dtype=torch.int, device=images.device)
        cp = torch.tensor([0, 0, 250], dtype=torch.int, device=images.device)
        L = torch.tensor([329288, ], dtype=torch.int, device=images.device)

        L_predict = L
        L_target = L

        reproduce_result = reproduce_loss(images, predict, lp, cp, L_predict, args)

        new_renderingloss = False
        if new_renderingloss:

            L_renderloss = torch.tensor([50, ], dtype=torch.int, device=images.device)
            renderloss_result_singleangle = 0
            renderloss_result_singleposition = 0

            for k in range(6):
                position_shift_lp_x = 0.5*target.shape[2]*(torch.rand([1], dtype=torch.float32) - 0.5) # [x,y]:  [-144, 144]
                position_shift_lp_y = 0.5*target.shape[3]*(torch.rand([1], dtype=torch.float32) - 0.5) # [x,y]
                position_shift_lp = torch.cat([position_shift_lp_x, position_shift_lp_y, torch.zeros([1], dtype=torch.float32)], dim=-1) # [x,y, 0]
                lp = lp + position_shift_lp.to(images.device)
                position_shift_cp_x = 0.5*target.shape[2]*(torch.rand([1], dtype=torch.float32) - 0.5) # [x,y]
                position_shift_cp_y = 0.5*target.shape[3]*(torch.rand([1], dtype=torch.float32) - 0.5) # [x,y]
                position_shift_cp = torch.cat([position_shift_cp_x, position_shift_cp_y, torch.zeros([1], dtype=torch.float32)], dim=-1) # [x,y, 0]
                cp = cp + position_shift_cp.to(images.device)

                im_gt = renderTex(target, lp, cp, L_target)
                im_pred = renderTex(predict, lp, cp, L_target)

                renderloss_result_singleposition = renderloss_result_singleposition + mse(torch.log(im_gt + 0.01), torch.log(im_pred + 0.01))

            print('\t renderloss_position: ', renderloss_result_singleposition/6)
            for k in range(3):
                single_angle = True
                renderloss_result_step = renderTex_loss(target, predict, L_renderloss, single_angle)
                renderloss_result_singleangle = renderloss_result_singleangle + renderloss_result_step[2]

            print('\t renderloss_angle: ', renderloss_result_singleangle/3)

            renderloss_result = (renderloss_result_singleposition+renderloss_result_singleangle)/9

            print('\t reproduce_result[1]: ', reproduce_result[1])
            print('\t renderloss_result[2]: ', renderloss_result)
            print('\t mse(target, predict): ', mse(target, predict))
            print('\t mse(light_target, light_predict): ', mse(L_target/1000000.0, L_predict/1000000.0))
            return [reproduce_result[0], im_gt, im_pred, renderloss_result+reproduce_result[1]+mse(L_target/1000000.0, L_predict/1000000.0)]

        else:
            predict = output_process(predict)
            target = output_process(target)
            rendering_loss_result = rendering_loss(target, predict, target.device, args)
            return [reproduce_result[0], rendering_loss_result[0], rendering_loss_result[1], rendering_loss_result[2]]
