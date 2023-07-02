from deepdrive_course.utils import torch_normalize_img, plot_to_pil_image
from captum.attr import LayerGradCam, LayerAttribution, GuidedGradCam
from captum.attr import visualization as viz
from einops import rearrange


def gradcam_analysis(img, label, model, layer):
    img = img.clone()
    gradcam = LayerGradCam(model, layer)
    attributions = gradcam.attribute(img.unsqueeze(0), label)
    upsampled_attributions = LayerAttribution.interpolate(attributions, img.shape[1:3])

    # Visualization
    img_normalized = torch_normalize_img(img)
    img_normalized_np = rearrange(img_normalized, "c h w -> h w c").numpy()
    upsampled_attributions_np = rearrange(upsampled_attributions, "1 c h w -> h w c").detach().cpu().numpy()

    fig, ax = viz.visualize_image_attr_multiple(
        upsampled_attributions_np,
        img_normalized_np,
        methods=["original_image", "heat_map", "blended_heat_map"],
        signs=["all", "positive", "positive"],
        titles=["Original", "Positive heatmap", "Positive blended"],
    )

    return plot_to_pil_image(figure=fig)


def guided_gradcam_analysis(img, label, model, layer):
    img = img.clone()
    guided_gradcam = GuidedGradCam(model, layer)
    attributions = guided_gradcam.attribute(img.unsqueeze(0), label)

    # Visualization
    img_normalized = torch_normalize_img(img)
    img_normalized_np = rearrange(img_normalized, "c h w -> h w c").numpy()
    attributions_np = rearrange(attributions, "1 c h w -> h w c").detach().cpu().numpy()

    fig, ax = viz.visualize_image_attr_multiple(
        attributions_np,
        img_normalized_np,
        methods=["original_image", "heat_map", "blended_heat_map"],
        signs=["all", "positive", "positive"],
        titles=["Original", "Positive heatmap", "Positive blended"],
    )

    return plot_to_pil_image(figure=fig)
