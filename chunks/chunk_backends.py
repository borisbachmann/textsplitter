class LinearChunker:
    def __init__(self, model, **specs):
        self.model = self._load_model(model)
        print(specs)
        pass

    def _load_model(self, model):
        return model

class GraphChunker:
    def __init__(self, model, **specs):
        self.model = self._load_modal(model)
        print(specs)

        def _load_model(self, model):
            return model
        pass


# Mapping of paragraph segmenter names to segmenter classes
CHUNK_SEGMENTER_MAP = {
    "linear": LinearChunker,
    "graph": GraphChunker
}
