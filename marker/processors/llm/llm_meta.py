from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

from marker.logger import get_logger
from tqdm import tqdm

from marker.processors.llm import BaseLLMSimpleBlockProcessor, BaseLLMProcessor
from marker.schema.document import Document
from marker.services import BaseService

logger = get_logger()


class LLMSimpleBlockMetaProcessor(BaseLLMProcessor):
    """
    A wrapper for simple LLM processors, so they can all run in parallel.
    """

    def __init__(
        self,
        processor_lst: List[BaseLLMSimpleBlockProcessor],
        llm_service: BaseService,
        config=None,
    ):
        super().__init__(llm_service, config)
        self.processors = processor_lst

    def __call__(self, document: Document):
        if not self.llm_enabled or self.llm_service is None:
            return

        total = sum(
            [len(processor.inference_blocks(document)) for processor in self.processors]
        )
        pbar = tqdm(
            desc="LLM processors running", disable=self.disable_tqdm, total=total
        )

        all_prompts = [
            processor.block_prompts(document) for processor in self.processors
        ]
        pending = []
        futures_map = {}
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            for i, prompt_lst in enumerate(all_prompts):
                for prompt in prompt_lst:
                    future = executor.submit(self.get_response, prompt)
                    pending.append(future)
                    futures_map[future] = {"processor_idx": i, "prompt_data": prompt}

            for future in pending:
                try:
                    result = future.result()
                    future_data = futures_map.pop(future)
                    processor: BaseLLMSimpleBlockProcessor = self.processors[
                        future_data["processor_idx"]
                    ]
                    # finalize the result
                    processor(result, future_data["prompt_data"], document)
                except Exception as e:
                    logger.warning(f"Error processing LLM response: {e}")

                pbar.update(1)

        pbar.close()

    def get_response(self, prompt_data: Dict[str, Any]):
        return self.llm_service(
            prompt_data["prompt"],
            prompt_data["image"],
            prompt_data["block"],
            prompt_data["schema"],
        )
