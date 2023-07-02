from captum.attr import GuidedGradCam, LayerAttribution, LayerGradCam, Occlusion
from captum.attr import visualization as viz
from deepdrive_course.utils import torch_normalize_img
from einops import rearrange


def visualize_attributions(attributions, img):
    img_normalized = torch_normalize_img(img)
    img_normalized_np = rearrange(img_normalized, "c h w -> h w c").numpy()
    attributions_np = rearrange(attributions, "1 c h w -> h w c").detach().cpu().numpy()

    return viz.visualize_image_attr_multiple(
        attributions_np,
        img_normalized_np,
        methods=["original_image", "heat_map", "blended_heat_map", "heat_map", "blended_heat_map"],
        signs=["all", "positive", "positive", "negative", "negative"],
        titles=["Original", "Positive heatmap", "Positive blended", "Negative heatmap", "Negative blended"],
        use_pyplot=False,
        fig_size=(10, 3.5),
    )


def gradcam_analysis(img, label, model, layer):
    img = img.clone()
    gradcam = LayerGradCam(model, layer)
    attributions = gradcam.attribute(img.unsqueeze(0), label)
    upsampled_attributions = LayerAttribution.interpolate(attributions, img.shape[1:3])

    return visualize_attributions(upsampled_attributions, img)


def guided_gradcam_analysis(img, label, model, layer):
    img = img.clone()
    guided_gradcam = GuidedGradCam(model, layer)
    attributions = guided_gradcam.attribute(img.unsqueeze(0), label)

    return visualize_attributions(attributions, img)


def occlusion_analysis(img, label, model, window=(3, 3, 3), strides=(3, 1, 1)):
    img = img.clone()
    ablator = Occlusion(model)
    attributions = ablator.attribute(img.unsqueeze(0), target=label, sliding_window_shapes=window, strides=strides)

    return visualize_attributions(attributions, img)
