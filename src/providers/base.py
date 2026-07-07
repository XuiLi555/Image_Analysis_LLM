"""Abstract provider interface.

A provider takes a text prompt plus one or more images and returns the
model's text response. Keeping this surface tiny makes it easy to add new
vendors (OpenAI, Google, a local server, ...) later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..image_utils import Image


class Provider(ABC):
    """Base class all vision-LLM backends implement."""

    def __init__(self, settings):
        self.settings = settings

    @abstractmethod
    def analyze(
        self,
        prompt: str,
        images: list[Image],
        system: str | None = None,
    ) -> str:
        """Send ``prompt`` and ``images`` to the model and return its text.

        Parameters
        ----------
        prompt:
            The user instruction / question.
        images:
            One or more images to include, in order.
        system:
            Optional system prompt steering the model's behaviour.
        """
        raise NotImplementedError
