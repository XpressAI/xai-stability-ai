from xai_components.base import InArg, OutArg, Component, xai_component
from stability_sdk import client
from stability_sdk.client import generation
from PIL import Image
import io
import warnings

@xai_component
class StabilityAPIInit(Component):
    """
    Component to initialize the Stability AI API Client.
    """
    STABILITY_HOST: InArg[str]
    STABILITY_KEY: InArg[str]

    stability_api_client: OutArg[client.StabilityInference]

    def execute(self, ctx) -> None:
        self.stability_api_client.value = client.StabilityInference(
            host=self.STABILITY_HOST.value,
            key=self.STABILITY_KEY.value,
            verbose=True,
        )


@xai_component
class GenerateImageFromPrompt(Component):
    """
    Component to generate an image from a text prompt using Stability AI.
    """
    stability_api_client: InArg[client.StabilityInference]
    prompt: InArg[str]
    seed: InArg[int]
    steps: InArg[int]

    generated_image: OutArg[Image.Image]

    def execute(self, ctx) -> None:
        answers = self.stability_api_client.value.generate(
            prompt=self.prompt.value,
            seed=self.seed.value,
            steps=self.steps.value,
        )

        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    warnings.warn(
                        "Your request activated the API's safety filters and could not be processed. "
                        "Please modify the prompt and try again."
                    )
                if artifact.type == generation.ARTIFACT_IMAGE:
                    self.generated_image.value = Image.open(io.BytesIO(artifact.binary))


@xai_component
class ModifyImageFromPrompt(Component):
    """
    Component to modify an image based on a prompt using Stability AI.
    """
    stability_api_client: InArg[client.StabilityInference]
    prompt: InArg[str]
    init_image: InArg[Image.Image]
    seed: InArg[int]
    start_schedule: InArg[float]
    steps: InArg[int]
    sampler: InArg[str]

    modified_image: OutArg[Image.Image]

    def execute(self, ctx) -> None:
        bytes_io = io.BytesIO()
        self.init_image.value.save(bytes_io, format='PNG')
        bytes_io.seek(0)

        answers = self.stability_api_client.value.generate(
            prompt=self.prompt.value,
            init_image=bytes_io,
            seed=self.seed.value,
            start_schedule=self.start_schedule.value,
            steps=self.steps.value,
            sampler=self.sampler.value,
        )

        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    warnings.warn(
                        "Your request activated the API's safety filters and could not be processed. "
                        "Please modify the prompt and try again."
                    )
                if artifact.type == generation.ARTIFACT_IMAGE:
                    self.modified_image.value = Image.open(io.BytesIO(artifact.binary))


@xai_component
class UpscaleImage(Component):
    """
    Component to upscale an image using Stability AI.
    """
    stability_api_client: InArg[client.StabilityInference]
    init_image: InArg[Image.Image]

    upscaled_image: OutArg[Image.Image]

    def execute(self, ctx) -> None:
        bytes_io = io.BytesIO()
        self.init_image.value.save(bytes_io, format='PNG')
        bytes_io.seek(0)

        answers = self.stability_api_client.value.upscale(
            init_image=bytes_io
        )

        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    warnings.warn(
                        "Your request activated the API's safety filters and could not be processed. "
                        "Please submit a different image and try again."
                    )
                if artifact.type == generation.ARTIFACT_IMAGE:
                    self.upscaled_image.value = Image.open(io.BytesIO(artifact.binary))
