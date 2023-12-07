from PIL import Image
import cv2
import numpy as np


from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InputField,
    invocation,
    InvocationContext,
    WithMetadata,
    WithWorkflow,
)

from invokeai.app.invocations.primitives import (
    ImageField,
    ImageOutput
)


@invocation(
    "clean_artifact_after_cut",
    title="Clean Image Artifacts After Cut",
    tags=["image", "clean"],
    category="image",
    version="1.0.0",
)
class CleanArtifactAfterCutInvocation(BaseInvocation, WithMetadata, WithWorkflow):
    """Remove artifacts from an image after separating it from the background"""
    image: ImageField = InputField(default=None, description="The image to be cleaned")
    iterations: int = InputField(default=5, description="The number of times morphological operations are applied")
    kernel_size: int = InputField(default=5, description="The size of the kernel used in morphological operations")
    threshold: float = InputField(default=0.5, description="The threshold for removing semi-transparent pixels: 0 for fully transparent to 1 for fully opaque")
    smooth_outline: int = InputField(default=5, description="The amount of blurring applied to the image's edges for a smoother outline")


    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)  
        cv_image = self.pil2cv2_image(image)
 
        original_alpha_channel = cv_image[:, :, 3]

        # Remove transparent pixels
        _, image_mask = cv2.threshold(
            src=original_alpha_channel, 
            thresh=int(255 * self.threshold), 
            maxval=255, 
            type=cv2.THRESH_BINARY
        )

        # Morphological closure operation
        image_mask = cv2.morphologyEx(
            src=image_mask, 
            op=cv2.MORPH_CLOSE, 
            kernel=(self.kernel_size, self.kernel_size),
            iterations=self.iterations
        )

        # Smooth outline
        blur_kernel_size = self.smooth_outline if self.smooth_outline % 2 == 1 else self.smooth_outline + 1
        image_mask = cv2.GaussianBlur(image_mask, (blur_kernel_size, blur_kernel_size), 0)

        # Set mask
        cv_image[:, :, 3] = cv2.bitwise_and(original_alpha_channel, image_mask)

        image_dto = context.services.images.create(
            image=self.cv2Pilimage(cv_image),
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            workflow=self.workflow,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )
    
    def pil2cv2_image(self, image):
        numpy_image = np.array(image)
        return cv2.cvtColor(numpy_image, cv2.COLOR_RGBA2BGRA)
    

    def cv2Pilimage(self, image):
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))