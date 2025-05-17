from typing import List

class VocabLinker:
    """
    This class is used to link the `query` to candidate entities in the `vocabulary`.
    """

    def __init__(self, vocab: List[str]):
        self.vocab = vocab

    def link(
            self, 
            query: str,
            top_k: int = 10,
        ) -> List[str]:
        """
        Link function
        """
        pass
