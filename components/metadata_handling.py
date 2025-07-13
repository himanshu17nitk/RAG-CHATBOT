from typing import List

class MetadataHandler:
    def __init__(self):
        pass

    def get_metadata(self, text: str, source: str, user_id: str) -> dict:
        return {"source": source, "user_id": user_id}
    
    def get_metadata_list(self, texts: List[str], source: str, user_id: str) -> List[dict]:
        return [self.get_metadata(text, source, user_id) for text in texts]
        