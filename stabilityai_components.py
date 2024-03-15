from xai_components.base import InArg, InCompArg, OutArg, Component, xai_component, secret
from stability_sdk import client
from stability_sdk.client import StabilityInference
from stability_sdk.client import generation

from PIL import Image

import io
import warnings

@xai_component
class StabilityAIAuthorize(Component):
    """
    Component to initialize the Stability AI API Client.
    """
    host: InArg[str]
    api_key: InCompArg[secret]
    from_env: InArg[bool]

    client: OutArg[StabilityInference]

    def execute(self, ctx) -> None:
        stability_client = client.StabilityInference(
            host=self.host.value if self.host.value is not None else 'grpc.stability.ai:443',
            key=self.api_key.value
        )

        ctx['stability_api'] = stability_client
        self.client.value = stability_client


@xai_component
class StabilityAIGenerateImage(Component):
    """
    Component to generate an image from a text prompt using Stability AI.
    """
    prompt: InCompArg[str]
    seed: InArg[int]
    steps: InArg[int]

    generated_image: OutArg[Image.Image]

    def execute(self, ctx) -> None:
        stability_api = ctx['stability_api']
        
        answers = stability_api.generate(
            prompt=self.prompt.value,
            seed=self.seed.value if self.seed.value is not None else 42,
            steps=self.steps.value if self.steps.value is not None else 50,
        )

        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    warnings.warn(
                        "Your request activated the API's safety filters and could not be processed. "
                        "Please modify the prompt and try again."
                    )
                    self.generated_image.value = None
                if artifact.type == generation.ARTIFACT_IMAGE:
                    self.generated_image.value = Image.open(io.BytesIO(artifact.binary))


@xai_component
class StabilityAIModifyImage(Component):
    """
    Component to modify an image based on a prompt using Stability AI.
    """
    prompt: InCompArg[str]
    init_image: InCompArg[Image]
    seed: InArg[int]
    start_schedule: InArg[float]
    steps: InArg[int]
    sampler: InArg[str]

    modified_image: OutArg[Image.Image]

    def execute(self, ctx) -> None:
        stability_api = ctx['stability_api']

        answers = stability_api.generate(
            prompt=self.prompt.value,
            init_image=self.init_image.value,
            seed=self.seed.value if self.seed.value is not None else 42,
            start_schedule=self.start_schedule.value if self.start_schedule.value is not None else 0.6,
            steps=self.steps.value if self.steps.value is not None else 50,
            sampler=self.sampler.value if self.sampler.value is not None else 'SAMPLER_DDIM',
        )

        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    warnings.warn(
                        "Your request activated the API's safety filters and could not be processed. "
                        "Please modify the prompt and try again."
                    )
                    self.modified_image.value = None

                if artifact.type == generation.ARTIFACT_IMAGE:
                    self.modified_image.value = Image.open(io.BytesIO(artifact.binary))


@xai_component
class StabilityAIUpscaleImage(Component):
    """
    Component to upscale an image using Stability AI.
    """
    init_image: InArg[Image.Image]

    upscaled_image: OutArg[Image.Image]

    def execute(self, ctx) -> None:
        stability_api = ctx['stability_api']

        answers = stability_api.upscale(
            init_image=self.init_image.value
        )

        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    warnings.warn(
                        "Your request activated the API's safety filters and could not be processed. "
                        "Please submit a different image and try again."
                    )
                    self.upscaled_image.value = None
                if artifact.type == generation.ARTIFACT_IMAGE:
                    self.upscaled_image.value = Image.open(io.BytesIO(artifact.binary))


@xai_component
class StabilityAIReadImageFile(Component):
    """
    Component to read an image file and output a PIL Image object.
    """
    file_path: InArg[str]
    image: OutArg[Image.Image]

    def execute(self, ctx) -> None:
        try:
            img = Image.open(self.file_path.value)
            self.image.value = img
        except Exception as e:
            raise Exception(f"Failed to read image file: {e}")


@xai_component
class StabilityAIWriteImage(Component):
    """
    Component to save a PIL Image object to a file.
    """
    image: InArg[Image.Image]
    save_path: InArg[str]

    def execute(self, ctx) -> None:
        try:
            self.image.value.save(self.save_path.value)
        except Exception as e:
            raise Exception(f"Failed to save image file: {e}")
